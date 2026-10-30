import sys, time, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.export import export_all
tag = sys.argv[1]; lpc = int(sys.argv[2]); Ts = tuple(int(x) for x in sys.argv[3].split(',')); out = sys.argv[4]
nf = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
t = time.time()
export_all(out, np.float16 if tag == "fp16" else np.float32, layers_per_chunk=lpc, Ts=Ts, nf=nf)
print("done", time.time() - t)
