import sys, time, numpy as np, onnx
from onnx import helper as h, TensorProto as T
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.npu import make_session
rng = np.random.default_rng(0)
W = rng.standard_normal((2048, 2048)).astype(np.float32) * 0.02
g = h.make_graph([h.make_node("MatMul", ["x", "W"], ["y"])], "g",
                 [h.make_tensor_value_info("x", T.FLOAT, [1, 1, 2048])],
                 [h.make_tensor_value_info("y", T.FLOAT, [1, 1, 2048])],
                 [h.make_tensor("W", T.FLOAT, W.shape, W.flatten())])
m = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); m.ir_version = 9
x = rng.standard_normal((1, 1, 2048)).astype(np.float32)
ref = x @ W
for npu in (False, True):
    s = make_session(m.SerializeToString(), npu=npu)
    print("providers:", s.get_providers())
    y = s.run(None, {"x": x})[0]
    t = time.perf_counter()
    for _ in range(50): s.run(None, {"x": x})
    print("npu" if npu else "cpu", "maxerr", np.abs(y - ref).max(), "ms", (time.perf_counter() - t) / 50 * 1e3)
