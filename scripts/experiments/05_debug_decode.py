import sys, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tokenizers import Tokenizer
from lfm3npu.engine import Engine, Cache
from lfm3npu.ref_numpy import RefModel, rms
from lfm3npu.config import *
tok = Tokenizer.from_file(os.path.join(ROOT, r"model_hf\tokenizer.json"))
ids = tok.encode("<|startoftext|><|im_start|>user\nBonjour, quelle est la capitale de la France ?<|im_end|>\n<|im_start|>assistant\n").ids
ref = RefModel(); E = ref.st.get("model.embed_tokens.weight"); ref.emb = E
_, rst = ref.forward(ids)
e = Engine(os.path.join(ROOT, "onnx_probe_fp16"), backend="npu", tag="fp16", log=lambda *a: None)
c = Cache(e.nf, e.dt); e.prefill(c, ids)
nxt = 124901
e.dbg.clear(); lg = e.decode(c, nxt)
# reference per chunk hidden
pos = np.array([len(ids)]); h = E[[nxt]]
for i in range(NL):
    h = ref.layer(i, h, pos, rst)
    if (i + 1) % 6 == 0:
        d = e.dbg[(i + 1) // 6 - 1]
        print(f"after layer {i}: rel err {np.linalg.norm(d - h[0]) / np.linalg.norm(h[0]):.4g}  ref absmax {np.abs(h).max():.3g}")
