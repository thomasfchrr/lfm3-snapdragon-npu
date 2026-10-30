"""Export the whole LFM3 as a set of static ONNX graphs: layer chunks (decode T=1 / prefill T=P) + lm_head."""
import os, json
import numpy as np
import onnx
from onnx import helper as h, TensorProto as TP
from .config import *
from .graphs import G, build_chunk, finish
from .weights import Safetensors


def build_head(st, dtype, rscale=1.0, vocab_split=1, quant=None, capture=False):
    """final embedding_norm + tied lm_head. Input hidden [1,1,D]; outputs logits parts [1,1,V/vocab_split]."""
    g = G(dtype, quant, capture)
    x = g.rms("hidden", st.get("model.embedding_norm.weight", np.float32), rscale)
    E = st.get("model.embed_tokens.weight", np.float32)           # [V,D]
    per = VOCAB // vocab_split
    outs = []
    for k in range(vocab_split):
        y = g.linear(x, E[k * per:(k + 1) * per], "head.in", "head.out", "head")
        outs.append((f"logits_{k}", y, [1, 1, per]))
    for key, t in g.caps:
        outs.append((f"cap_{key}", t, None))
    return finish(g, [("hidden", [1, 1, D])], outs, "head")


def export_all(outdir, dtype=np.float16, layers_per_chunk=6, Ts=(64,), nf=2048, rscale=1.0, vocab_split=1, log=print,
               quant=None, capture=False, tag=None):
    os.makedirs(outdir, exist_ok=True)
    st = Safetensors(f"{MODEL_DIR}/model.safetensors")
    tag = tag or ("fp16" if dtype == np.float16 else "fp32")
    manifest = {"dtype": tag, "layers_per_chunk": layers_per_chunk, "T_prefill": max(Ts), "Ts": list(Ts), "nf": nf,
                "rscale": rscale, "io": "fp16" if dtype == np.float16 else "fp32", "quant": bool(quant), "vocab_split": vocab_split, "chunks": [], "head": []}
    for l0 in range(0, NL, layers_per_chunk):
        l1 = min(NL, l0 + layers_per_chunk)
        for T in Ts:
            fn = f"chunk_{l0:02d}_{l1:02d}_T{T}_{tag}.onnx"
            path = os.path.join(outdir, fn)
            if not os.path.exists(path):
                m = build_chunk(st, l0, l1, T, nf, dtype=dtype, rscale=rscale, quant=quant, capture=capture)
                onnx.save(m, path)
                log(f"wrote {fn} ({os.path.getsize(path) / 1e6:.0f} MB)")
        manifest["chunks"].append([l0, l1])
    for k in range(1):
        fn = f"head_{tag}.onnx"
        path = os.path.join(outdir, fn)
        if not os.path.exists(path):
            m = build_head(st, dtype, rscale, vocab_split, quant, capture)
            onnx.save(m, path)
            log(f"wrote {fn} ({os.path.getsize(path) / 1e6:.0f} MB)")
    with open(os.path.join(outdir, f"manifest_{tag}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
