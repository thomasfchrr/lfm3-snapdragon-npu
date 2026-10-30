"""Teacher-forced quality eval on held-out text. usage: 12_eval_quality.py NAME BACKEND TAG ONNX_DIR
NAME=ref  -> store fp32 reference top-50 logprobs; other names are compared to it (ppl, top-1 agreement, KL on top-50)."""
import sys, json, os, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.engine import Engine, Cache
from lfm3npu.corpus import eval_texts

R = ROOT
name, backend, tag, d = sys.argv[1:5]
e = Engine(d, backend=backend, tag=tag, log=lambda *a: None)
texts = eval_texts()
rec = {"true_lp": [], "top1": [], "topk_id": [], "topk_lp": []}
t0 = time.time(); steps = 0
for ids in texts:
    c = Cache(e.nf, e.dt)
    lg = e.prefill(c, ids[:4])
    for j in range(4, len(ids) + 0):
        lp = lg - lg.max(); lp = lp - np.log(np.exp(lp).sum())
        top = np.argsort(-lp)[:50]
        rec["true_lp"].append(float(lp[ids[j]])); rec["top1"].append(int(top[0]))
        rec["topk_id"].append(top); rec["topk_lp"].append(lp[top])
        if j < len(ids):
            lg = e.decode(c, ids[j])
        steps += 1
dt = time.time() - t0
true_lp = np.array(rec["true_lp"]); top1 = np.array(rec["top1"])
topk_id = np.stack(rec["topk_id"]); topk_lp = np.stack(rec["topk_lp"])
res = {"name": name, "tokens": int(len(true_lp)), "ppl": float(np.exp(-true_lp.mean())), "sec": dt}
rp = R + r"\results\quality_ref.npz"
if name == "ref":
    np.savez(rp, true_lp=true_lp, top1=top1, topk_id=topk_id, topk_lp=topk_lp)
else:
    ref = np.load(rp)
    res["top1_agree"] = float((ref["top1"] == top1).mean())
    kls = []
    for i in range(len(true_lp)):
        ids_r, lp_r = ref["topk_id"][i], ref["topk_lp"][i]
        # quant logprobs on reference support are not stored; approximate with top-50 overlap mass
        m = {int(a): b for a, b in zip(topk_id[i], topk_lp[i])}
        p = np.exp(lp_r); p /= p.sum()
        q = np.array([np.exp(m.get(int(a), lp_r[-1] - 3.0)) for a in ids_r]); q /= q.sum()
        kls.append(float((p * (np.log(p) - np.log(q))).sum()))
    res["kl_top50"] = float(np.mean(kls))
    res["mean_abs_dlogp_true"] = float(np.abs(true_lp - ref["true_lp"]).mean())
print(json.dumps(res))
qf = R + r"\results\quality.json"
allr = json.load(open(qf)) if os.path.exists(qf) else {}
allr[name] = res
json.dump(allr, open(qf, "w"), indent=1)
