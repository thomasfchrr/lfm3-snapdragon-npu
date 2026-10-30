"""Host-side runtime: owns tokens/embeddings/rope/masks/caches/sampling; every transformer FLOP runs inside ONNX
sessions (QNN HTP = Hexagon NPU, or CPU EP for the baseline)."""
import json, os, time
import numpy as np
from .config import *
from .graphs import NEG
from .npu import make_session
from .weights import Safetensors


class Cache:
    def __init__(self, nf, dt):
        self.dt = dt
        self.p = 0                                   # next absolute position
        self.conv = {i: np.zeros((1, D, KC), dt) for i in range(NL) if LT[i] == "conv"}
        self.k, self.v, self.n, self.cpos = {}, {}, {}, {}
        for i in range(NL):
            if LT[i] == "conv":
                continue
            nc = nf if LT[i] == "full_attention" else WIN
            self.k[i] = np.zeros((1, NKV, nc, HD), dt)
            self.v[i] = np.zeros((1, NKV, nc, HD), dt)
            self.n[i] = 0
            if LT[i] == "sliding_attention":
                self.cpos[i] = np.full(WIN, -1, np.int64)


class Engine:
    def __init__(self, onnx_dir, backend="npu", tag="fp16", log=print, htp_opts=None, ctx_dir=None):
        self.backend, self.tag = backend, tag
        with open(os.path.join(onnx_dir, f"manifest_{tag}.json")) as f:
            self.mf = json.load(f)
        self.dt = np.float16 if self.mf.get("io", tag) == "fp16" else np.float32
        self.cap_cb = None
        self.nf = self.mf["nf"]
        self.Ts = sorted(self.mf.get("Ts", [self.mf["T_prefill"]]))
        self.TP = max(self.Ts)
        st = Safetensors(f"{MODEL_DIR}/model.safetensors")
        t0 = time.time()
        self.E = st.get("model.embed_tokens.weight", self.dt)               # token embedding table (host gather)
        log(f"embeddings loaded {time.time() - t0:.1f}s {self.E.shape} {self.E.dtype}")
        self.sess = {}
        npu = backend == "npu"
        # NPU: ONE static graph size serves prefill AND decode (right-padded, `valid` mask) because HTP has a ~4 GB
        # address space shared by every session of the process. CPU baseline may load several sizes (T=1 decode).
        for l0, l1 in self.mf["chunks"]:
            for T in self.Ts:
                t0 = time.time()
                fn = os.path.join(onnx_dir, f"chunk_{l0:02d}_{l1:02d}_T{T}_{tag}.onnx")
                self.sess[(l0, T)] = make_session(fn, npu=npu, fp16=True, extra_opts=htp_opts)
                log(f"session chunk {l0}-{l1} T={T} ready {time.time() - t0:.1f}s")
        t0 = time.time()
        self.head = make_session(os.path.join(onnx_dir, f"head_{tag}.onnx"), npu=npu, fp16=True, extra_opts=htp_opts)
        log(f"session head ready {time.time() - t0:.1f}s")
        self.timing = {}
        self.dbg = []

    # ---- host helpers
    def _masks(self, cache, n, T, pos):
        """pos: absolute positions of real tokens (n); rows [0,T-n) are left padding."""
        out = {}
        pad = T - n
        r = np.arange(T)
        real = r < n
        pr = np.zeros(T, np.int64); pr[:n] = pos
        nfull = [cache.n[i] for i in cache.n if LT[i] == "full_attention"]
        nfull = nfull[0] if nfull else 0
        mf = np.full((T, self.nf + T), NEG, np.float32)
        mf[:, :nfull] = 0
        tri = (r[None, :] <= r[:, None]) & real[None, :] & real[:, None]
        mf[:, self.nf:][tri] = 0
        out["mask_full"] = np.tile(mf, (NH // NKV, 1))[None, None].astype(self.dt)
        # sliding: all sliding layers share cpos (updated together) -> take first
        sl = [i for i in cache.cpos]
        if sl:
            cp = cache.cpos[sl[0]]
            ms = np.full((T, WIN + T), NEG, np.float32)
            okc = (cp[None, :] >= 0) & (cp[None, :] > pr[:, None] - WIN)
            ms[:, :WIN][okc & real[:, None]] = 0
            tri2 = tri & (pr[None, :] > pr[:, None] - WIN)
            ms[:, WIN:][tri2] = 0
            out["mask_sl"] = np.tile(ms, (NH // NKV, 1))[None, None].astype(self.dt)
        return out

    def _rope(self, T, n, pos):
        pr = np.zeros(T, np.int64); pr[:n] = pos
        out = {}
        for kind, key in (("full", "full"), ("sliding", "sl")):
            c, s = rope_tables(pr, kind)
            out[f"cos_{key}"] = c[None, None].astype(self.dt)
            out[f"sin_{key}"] = s[None, None].astype(self.dt)
        return out

    def _run_chunks(self, cache, ids, T):
        n = len(ids)
        pad = T - n
        pos = np.arange(cache.p, cache.p + n)
        hid = np.zeros((1, T, D), self.dt)
        hid[0, :n] = self.E[np.asarray(ids)]
        valid = np.zeros((1, T, 1), self.dt); valid[0, :n] = 1
        csel = np.zeros((KC + T, KC), self.dt); csel[n + np.arange(KC), np.arange(KC)] = 1
        shared = {"valid": valid, "csel": csel}
        shared.update(self._rope(T, n, pos))
        shared.update(self._masks(cache, n, T, pos))
        for l0, l1 in self.mf["chunks"]:
            s = self.sess[(l0, T)]
            feeds = {"hidden": hid}
            for inp in s.get_inputs():
                nm = inp.name
                if nm in feeds:
                    continue
                if nm in shared:
                    feeds[nm] = shared[nm]
                elif nm.startswith("cstate_"):
                    feeds[nm] = cache.conv[int(nm.split("_")[1])]
                elif nm.startswith("pk_"):
                    feeds[nm] = cache.k[int(nm.split("_")[1])]
                elif nm.startswith("pv_"):
                    feeds[nm] = cache.v[int(nm.split("_")[1])]
            t0 = time.perf_counter()
            outs = s.run(None, feeds)
            self.timing[(l0, T)] = self.timing.get((l0, T), 0) + time.perf_counter() - t0
            om = {o.name: a for o, a in zip(s.get_outputs(), outs)}
            hid = om["hidden_out"]
            self.dbg.append(hid[0, n - 1].astype(np.float32).copy())
            for nm, a in om.items():
                if nm.startswith("cap_"):
                    if self.cap_cb:
                        self.cap_cb(nm[4:], a[0, :n])
                elif nm.startswith("cstate_"):
                    cache.conv[int(nm.split("_")[1])] = a
                elif nm.startswith("nk_") or nm.startswith("nv_"):
                    i = int(nm.split("_")[1])
                    buf = cache.k[i] if nm[1] == "k" else cache.v[i]
                    new = a[:, :, :n, :]
                    if LT[i] == "full_attention":
                        if cache.n[i] + n > self.nf:
                            raise RuntimeError("context window of the NPU graph is full")
                        buf[:, :, cache.n[i]:cache.n[i] + n, :] = new
                    else:
                        buf[:, :, :WIN - n, :] = buf[:, :, n:, :].copy()
                        buf[:, :, WIN - n:, :] = new
        # bookkeeping of counters (once per call)
        for i in cache.n:
            if LT[i] == "full_attention":
                cache.n[i] += n
            else:
                cache.cpos[i][:WIN - n] = cache.cpos[i][n:].copy()
                cache.cpos[i][WIN - n:] = pos
                cache.n[i] = min(WIN, cache.n[i] + n)
        cache.p += n
        return hid[:, n - 1:n, :]

    def prefill(self, cache, ids):
        ids = list(ids); i = 0; last = None
        while i < len(ids):
            part = ids[i:i + self.TP]
            last = self._run_chunks(cache, part, min(t for t in self.Ts if t >= len(part)))
            i += len(part)
        return self.logits(last)

    def decode(self, cache, tok):
        h = self._run_chunks(cache, [tok], self.Ts[0])
        return self.logits(h)

    def logits(self, hid_last):
        t0 = time.perf_counter()
        res = self.head.run(None, {"hidden": hid_last.astype(self.dt)})
        self.timing["head"] = self.timing.get("head", 0) + time.perf_counter() - t0
        names = [o.name for o in self.head.get_outputs()]
        outs = []
        for nm, a in zip(names, res):
            if nm.startswith("cap_"):
                if self.cap_cb:
                    self.cap_cb(nm[4:], a.reshape(-1, a.shape[-1]))
            else:
                outs.append(a)
        return np.concatenate([o.reshape(-1) for o in outs]).astype(np.float32)
