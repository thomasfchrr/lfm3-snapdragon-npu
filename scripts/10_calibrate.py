"""Collect activation ranges of every quantized MatMul input/output by running fp32 capture graphs on CPU."""
import sys, json, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.export import export_all
from lfm3npu.engine import Engine, Cache
from lfm3npu.corpus import calibration_texts

R = ROOT
cal_dir = R + r"\onnx_cal"
Tcal = 32
export_all(cal_dir, np.float32, layers_per_chunk=6, Ts=(Tcal,), capture=True,
           quant={"scales": {}, "default_bits": 8} if False else None, tag="fp32")
e = Engine(cal_dir, backend="cpu", tag="fp32", log=lambda *a: None)
stats = {}


def cb(key, a):
    a = np.abs(a.astype(np.float32))
    st = stats.setdefault(key, {"amax": 0.0, "p999": [], "p9999": [], "n": 0})
    st["amax"] = max(st["amax"], float(a.max()))
    flat = a.reshape(-1)
    st["p999"].append(float(np.percentile(flat, 99.9))); st["p9999"].append(float(np.percentile(flat, 99.99)))
    st["n"] += 1


e.cap_cb = cb
texts = calibration_texts()
print(len(texts), "sequences,", sum(len(t) for t in texts), "tokens", flush=True)
t0 = time.time()
for k, ids in enumerate(texts):
    c = Cache(e.nf, e.dt)
    split = max(len(ids) - 16, 1)
    e.prefill(c, ids[:split])
    for tkn in ids[split:]:            # teacher-forced decode steps also feed the lm_head statistics
        e.decode(c, tkn)
    print(f"seq {k + 1}/{len(texts)} len {len(ids)} elapsed {time.time() - t0:.0f}s", flush=True)
out = {k: {"amax": v["amax"], "p999": float(np.max(v["p999"])), "p9999": float(np.max(v["p9999"])),
           "p9999_mean": float(np.mean(v["p9999"])), "n": v["n"]} for k, v in stats.items()}
json.dump(out, open(R + r"\results\calib_stats.json", "w"), indent=1)
print("saved", len(out), "tensors")
