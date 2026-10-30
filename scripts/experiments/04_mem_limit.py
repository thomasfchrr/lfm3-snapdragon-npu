"""How many fp16 chunk graphs can the HTP hold at once? Load sessions one by one and run each after every load."""
import sys, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lfm3npu.npu import make_session
D = os.path.join(ROOT, "onnx_probe_fp16")
names = sys.argv[1:]
sess = []
def feed(s):
    return {i.name: np.zeros(i.shape, np.float16) for i in s.get_inputs()}
for n in names:
    t = time.time()
    s = make_session(f"{D}/{n}", npu=True)
    sess.append((n, s))
    ok = []
    for nn, ss in sess:
        try:
            ss.run(None, feed(ss)); ok.append("ok")
        except Exception as ex:
            ok.append("FAIL:" + str(ex)[:80].replace("\n", " "))
    print(f"loaded {n} ({time.time()-t:.1f}s)  ->", ok, flush=True)
