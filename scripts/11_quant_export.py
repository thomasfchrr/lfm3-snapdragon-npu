"""Build quantized (A16Wx) static graphs from calibration stats.
usage: 11_quant_export.py TAG STAT MARGIN WBITS T[,T2] [io=fp32|fp16] [mlp_bits]
  STAT: amax | p9999 | p999 ; scale = MARGIN * STAT"""
import sys, json, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.export import export_all

R = ROOT
tag, stat, margin, wbits, Ts = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), tuple(int(x) for x in sys.argv[5].split(","))
io = sys.argv[6] if len(sys.argv) > 6 else "fp32"
mlp_bits = int(sys.argv[7]) if len(sys.argv) > 7 else wbits
cs = json.load(open(R + r"\results\calib_stats.json"))
scales = {k: max(margin * v[stat], 1e-4) for k, v in cs.items()}
quant = {"scales": scales, "default_bits": wbits, "bits": {"w1": mlp_bits, "w2": mlp_bits, "w3": mlp_bits}}
export_all(R + rf"\onnx_{tag}", np.float16 if io == "fp16" else np.float32, layers_per_chunk=6, Ts=Ts, quant=quant, tag=tag)
print("exported", tag)
