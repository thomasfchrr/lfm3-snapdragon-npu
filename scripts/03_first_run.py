import sys, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer
from lfm3npu.engine import Engine, Cache
from lfm3npu.ref_numpy import RefModel
tok = Tokenizer.from_file(os.path.join(ROOT, r"model_hf\tokenizer.json"))
ids = tok.encode("<|startoftext|><|im_start|>user\nBonjour, quelle est la capitale de la France ?<|im_end|>\n<|im_start|>assistant\n").ids
rp = os.path.join(ROOT, r"results\ref_logits_p1.npy")
if not os.path.exists(rp):
    np.save(rp, RefModel().forward(ids)[0])
ref = np.load(rp)
backend = sys.argv[1]; tag = sys.argv[2]
e = Engine(sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "onnx_w8b"), backend=backend, tag=tag)
c = Cache(e.nf, e.dt)
t = time.time(); lg = e.prefill(c, ids); print("prefill", round(time.time() - t, 3), "s")
print("corr", np.corrcoef(lg, ref)[0, 1], "max|d|", np.abs(lg - ref).max(), "top ref", np.argsort(-ref)[:5], "top npu", np.argsort(-lg)[:5])
out = []
t = time.time()
for _ in range(24):
    nx = int(np.argmax(lg)); out.append(nx)
    lg = e.decode(c, nx)
dt = time.time() - t
print(f"decode {len(out)/dt:.2f} tok/s ({dt/len(out)*1e3:.1f} ms/tok)")
print(repr(tok.decode(out, skip_special_tokens=False)))
print({k: round(v, 3) for k, v in e.timing.items()})
