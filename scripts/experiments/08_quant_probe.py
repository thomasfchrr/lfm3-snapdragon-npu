"""Probe: does QNN HTP accept (a) weight-only int8/int4 DequantizeLinear->MatMul(fp16), (b) fully quantized A16W8 MatMul?
Measures latency of a single [1,8,2048]x[2048,7680] linear x 8 (memory-bound) against the fp16 baseline."""
import sys, time, numpy as np, onnx
from onnx import helper as h, TensorProto as TP, numpy_helper as nh
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from lfm3npu.npu import make_session

mode = sys.argv[1]
K, N, T, L = 2048, 7680, 8, 8
rng = np.random.default_rng(0)
Ws = [(rng.standard_normal((K, N)) * 0.02).astype(np.float32) for _ in range(L)]
x_ref = rng.standard_normal((1, T, K)).astype(np.float32)
nodes, inits = [], []
cur = "x"
dt = TP.FLOAT if mode.startswith("a16") else TP.FLOAT16
for i, W in enumerate(Ws):
    if mode == "fp16":
        inits.append(nh.from_array(W.astype(np.float16), f"w{i}"))
        nodes.append(h.make_node("MatMul", [cur, f"w{i}"], [f"y{i}"]))
    elif mode in ("w8", "w4"):
        qmax = 127 if mode == "w8" else 7
        sc = np.abs(W).max(0) / qmax                         # per output channel (axis=1)
        q = np.clip(np.round(W / sc), -qmax - 1, qmax).astype(np.int8)
        if mode == "w4":
            inits.append(h.make_tensor(f"w{i}", TP.INT4, W.shape, q.flatten().tolist()))
        else:
            inits.append(nh.from_array(q, f"w{i}"))
        inits.append(nh.from_array(sc.astype(np.float16), f"s{i}"))
        nodes.append(h.make_node("DequantizeLinear", [f"w{i}", f"s{i}"], [f"wd{i}"], axis=1))
        nodes.append(h.make_node("MatMul", [cur, f"wd{i}"], [f"y{i}"]))
    if mode.startswith("a16"):
        wbits = 4 if mode.endswith("w4") else 8
        qmax = 127 if wbits == 8 else 7
        sc = np.abs(W).max(0) / qmax
        q = np.clip(np.round(W / sc), -qmax - 1, qmax).astype(np.int8)
        if wbits == 4:
            inits.append(h.make_tensor(f"w{i}", TP.INT4, W.shape, q.flatten().tolist()))
        else:
            inits.append(nh.from_array(q, f"w{i}"))
        inits.append(nh.from_array(sc.astype(np.float32), f"s{i}"))
        inits.append(nh.from_array(np.array(8.0 / 32768, np.float32), f"sx{i}")); inits.append(nh.from_array(np.array(32768, np.uint16), f"zx{i}"))
        inits.append(nh.from_array(np.array(16.0 / 32768, np.float32), f"sy{i}")); inits.append(nh.from_array(np.array(32768, np.uint16), f"zy{i}"))
        nodes.append(h.make_node("QuantizeLinear", [cur, f"sx{i}", f"zx{i}"], [f"xq{i}"]))
        nodes.append(h.make_node("DequantizeLinear", [f"xq{i}", f"sx{i}", f"zx{i}"], [f"xd{i}"]))
        nodes.append(h.make_node("DequantizeLinear", [f"w{i}", f"s{i}"], [f"wd{i}"], axis=1))
        nodes.append(h.make_node("MatMul", [f"xd{i}", f"wd{i}"], [f"ym{i}"]))
        nodes.append(h.make_node("QuantizeLinear", [f"ym{i}", f"sy{i}", f"zy{i}"], [f"yq{i}"]))
        nodes.append(h.make_node("DequantizeLinear", [f"yq{i}", f"sy{i}", f"zy{i}"], [f"y{i}"]))
    cur = f"y{i}"
    if i < L - 1:   # keep shapes consistent: slice back to K features
        inits.append(nh.from_array(np.array([0], np.int64), f"a{i}")); inits.append(nh.from_array(np.array([K], np.int64), f"b{i}"))
        inits.append(nh.from_array(np.array([2], np.int64), f"c{i}"))
        nodes.append(h.make_node("Slice", [cur, f"a{i}", f"b{i}", f"c{i}"], [f"z{i}"])); cur = f"z{i}"
g = h.make_graph(nodes, "g", [h.make_tensor_value_info("x", dt, [1, T, K])], [h.make_tensor_value_info(cur, dt, None)], inits)
m = h.make_model(g, opset_imports=[h.make_opsetid("", 21)]); m.ir_version = 10
onnx.save(m, os.path.join(ROOT, r"results\_probe.onnx"))
try:
    s = make_session(os.path.join(ROOT, r"results\_probe.onnx"), npu=True)
except Exception as ex:
    print(mode, "SESSION FAIL", str(ex)[:300]); sys.exit()
x = x_ref.astype(np.float32 if mode.startswith("a16") else np.float16)
for _ in range(3): s.run(None, {"x": x})
t = time.perf_counter(); n = 30
for _ in range(n): s.run(None, {"x": x})
dtm = (time.perf_counter() - t) / n * 1e3
wb = L * K * N * {"fp16": 2, "w8": 1, "w4": 0.5, "a16w8": 1, "a16w4": 0.5}[mode] / 1e6
print(f"{mode}: {dtm:.2f} ms/run for {wb:.0f} MB of weights -> {wb / dtm:.1f} GB/s effective")
