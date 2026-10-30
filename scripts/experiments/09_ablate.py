"""Where does chunk time go? time chunk 0-6 (T=8) with variants."""
import sys, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lfm3npu import graphs
from lfm3npu.weights import Safetensors
from lfm3npu.npu import make_session
from lfm3npu.config import *
st = Safetensors(f"{MODEL_DIR}/model.safetensors")
def bench(name, l0, l1, nf, abl=(), T=8):
    graphs.ABL.clear(); graphs.ABL.update(abl)
    m = graphs.build_chunk(st, l0, l1, T, nf)
    s = make_session(m.SerializeToString(), npu=True)
    feeds = {i.name: np.zeros(i.shape, np.float16) for i in s.get_inputs()}
    for _ in range(5): s.run(None, feeds)
    t = time.perf_counter(); n = 40
    for _ in range(n): s.run(None, feeds)
    print(f"{name:34s} {(time.perf_counter()-t)/n*1e3:6.2f} ms", flush=True)
bench("layers0-6 nf=2048", 0, 6, 2048)
bench("layers0-6 nf=512", 0, 6, 512)
bench("layers0-6 nf=2048 no-norm", 0, 6, 2048, ("norm",))
bench("conv only (layers 0,1) nf=2048", 0, 2, 2048)
bench("layers 2-4 (attn+conv+attn) nf=2048", 2, 5, 2048)
bench("layers 2-4 nf=256", 2, 5, 256)
