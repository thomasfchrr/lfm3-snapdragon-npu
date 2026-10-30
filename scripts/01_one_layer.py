"""Build 1 layer at T tokens, check NPU strict + accuracy vs numpy ref.
usage: 01_one_layer.py LAYER T fp16|fp32 npu|cpu"""
import sys, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.config import *
from lfm3npu.graphs import build_chunk, NEG
from lfm3npu.weights import Safetensors
from lfm3npu.ref_numpy import RefModel
from lfm3npu.npu import make_session

layer = int(sys.argv[1]); T = int(sys.argv[2]); dt = np.float16 if sys.argv[3] == "fp16" else np.float32
npu = sys.argv[4] == "npu"
NF = 256
st = Safetensors(f"{MODEL_DIR}/model.safetensors")
rng = np.random.default_rng(1)
h0 = (rng.standard_normal((T, D)) * 1.0).astype(np.float32)
pos = np.arange(T)
ref = RefModel(st); state = {}
href = ref.layer(layer, h0.copy(), pos, state)

m = build_chunk(st, layer, layer + 1, T, NF, dtype=dt)
t = time.time()
sess = make_session(m.SerializeToString(), npu=npu, fp16=True)
print("session built", round(time.time() - t, 1), "s", sess.get_providers())
names = [i.name for i in sess.get_inputs()]
feeds = {"hidden": h0[None].astype(dt), "valid": np.ones((1, T, 1), dt)}
for kind in ("full", "sl"):
    c, s = rope_tables(pos, "full" if kind == "full" else "sliding")
    feeds[f"cos_{kind}"] = c[None, None].astype(dt); feeds[f"sin_{kind}"] = s[None, None].astype(dt)
    nc = NF if kind == "full" else WIN
    msk = np.full((T, nc + T), NEG, np.float32)
    for r in range(T):
        lo = max(0, r + 1 - WIN) if kind == "sl" else 0
        msk[r, nc + lo:nc + r + 1] = 0
    feeds[f"mask_{kind}"] = np.tile(msk, (NH // NKV, 1))[None, None].astype(dt)
feeds = {k: v for k, v in feeds.items() if k in names}
for i in sess.get_inputs():
    if i.name not in feeds:
        feeds[i.name] = np.zeros(i.shape, dt)
out = sess.run(None, feeds)
y = out[0][0].astype(np.float32)
err = np.abs(y - href); rel = np.linalg.norm(y - href) / np.linalg.norm(href)
print(f"layer {layer} T={T} {dt.__name__} npu={npu}: max|err|={err.max():.4g} rel={rel:.4g} ref_absmax={np.abs(href).max():.3g}")
t = time.perf_counter(); n = 20
for _ in range(n): sess.run(None, feeds)
print("ms/run", round((time.perf_counter() - t) / n * 1e3, 2))
