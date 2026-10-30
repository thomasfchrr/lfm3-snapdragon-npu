"""Throughput benchmark. usage: 13_bench.py NAME BACKEND TAG ONNX_DIR
Measures: load time, prefill tok/s for several prompt lengths, decode tok/s (greedy) at several context depths,
per-component time (chunks vs head vs host), and process memory. Writes results/bench_NAME.json."""
import sys, json, time, os, numpy as np, psutil
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = ROOT
name, backend, tag, d = sys.argv[1:5]
from lfm3npu.engine import Engine, Cache
from lfm3npu import chat

t0 = time.time()
e = Engine(d, backend=backend, tag=tag, log=lambda *a: None)
load_s = time.time() - t0
rng = np.random.default_rng(0)
res = {"name": name, "backend": backend, "tag": tag, "dir": os.path.basename(d), "load_s": load_s, "Ts": e.Ts,
       "rss_mb": psutil.Process().memory_info().rss / 1e6}
toks = lambda n: list(rng.integers(1000, 100000, n))

# warmup
c = Cache(e.nf, e.dt); e.prefill(c, toks(16)); [e.decode(c, 1234) for _ in range(4)]

res["prefill"] = {}
for n in (16, 64, 256, 1024):
    if n > e.nf - 64:
        continue
    best = 1e9
    for _ in range(2):
        c = Cache(e.nf, e.dt); t = time.perf_counter(); e.prefill(c, toks(n)); best = min(best, time.perf_counter() - t)
    res["prefill"][n] = {"s": best, "tok_s": n / best}
    print("prefill", n, f"{best:.3f}s {n / best:.1f} tok/s", flush=True)

res["decode"] = {}
for depth in (32, 512, 1536):
    if depth > e.nf - 80:
        continue
    c = Cache(e.nf, e.dt); e.prefill(c, toks(depth))
    e.timing.clear()
    t = time.perf_counter(); n = 48
    for _ in range(n):
        lg = e.decode(c, int(np.argmax(rng.standard_normal(10))) + 1000)
    dt = time.perf_counter() - t
    chunk_t = sum(v for k, v in e.timing.items() if k != "head") / n
    res["decode"][depth] = {"ms_per_tok": dt / n * 1e3, "tok_s": n / dt, "chunks_ms": chunk_t * 1e3,
                            "head_ms": e.timing.get("head", 0) / n * 1e3}
    print("decode @ctx", depth, f"{n / dt:.2f} tok/s ({dt / n * 1e3:.1f} ms; chunks {chunk_t * 1e3:.1f} head {e.timing.get('head', 0) / n * 1e3:.1f})", flush=True)
res["rss_mb_end"] = psutil.Process().memory_info().rss / 1e6
json.dump(res, open(R + rf"\results\bench_{name}.json", "w"), indent=1)
