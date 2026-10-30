"""Hand-written static ONNX graph builder for LFM3 layer chunks (NPU-friendly: static shapes, explicit state I/O,
host-managed caches, no Gather/Scatter/If/Loop in the transformer body).

Two numeric modes:
  * float   : fp16/fp32 weights, QNN runs it in fp16 (weights cost 2 bytes/param of DRAM traffic)
  * quant   : A16W8 / A16W4 QDQ around every big MatMul (uint16 activations, per-channel int8/int4 weights);
              everything else (norms, RoPE, softmax, gating) stays float. Scales come from calibration.
"""
import numpy as np
import onnx
from onnx import helper as h, TensorProto as TP
from .config import *
from .weights import Safetensors

NEG = -30000.0
ABL = set()          # debug ablations: "norm" (benchmark only, changes numerics)
BIG = 2**31 - 1
ZP = 32768           # uint16 symmetric zero point


class G:
    def __init__(self, dtype, quant=None, capture=False):
        self.dt = dtype
        self.tp = TP.FLOAT16 if dtype == np.float16 else TP.FLOAT
        self.nodes, self.inits, self.n = [], [], 0
        self.quant, self.capture = quant, capture
        self.caps = []                 # (cap_name, tensor_name) for calibration outputs
        self._qin = {}                 # memo: (key, tensor) -> dequantized tensor

    def name(self, p="t"):
        self.n += 1
        return f"{p}{self.n}"

    def const(self, arr, keep_dtype=False):
        name = self.name("c")
        arr = np.asarray(arr)
        if not keep_dtype and arr.dtype.kind == "f":
            arr = arr.astype(self.dt)
        self.inits.append(onnx.numpy_helper.from_array(arr, name))
        return name

    def op(self, typ, ins, nout=1, **attrs):
        outs = [self.name() for _ in range(nout)]
        self.nodes.append(h.make_node(typ, ins, outs, **attrs))
        return outs[0] if nout == 1 else outs

    def ints(self, v):
        return self.const(np.asarray(v, np.int64), keep_dtype=True)

    def scalar(self, v):
        return self.const(np.asarray(v, np.float32))

    # ---- blocks
    def rms(self, x, w, rscale=1.0):
        """RMSNorm over last axis. rscale pre-scales x to dodge fp16 overflow in x^2 (eps rescaled accordingly)."""
        if "norm" in ABL:
            return self.op("Mul", [x, self.const(w)])
        xs = self.op("Mul", [x, self.scalar(rscale)]) if rscale != 1.0 else x
        var = self.op("ReduceMean", [self.op("Mul", [xs, xs]), self.ints([-1])], keepdims=1)
        den = self.op("Sqrt", [self.op("Add", [var, self.scalar(EPS * rscale * rscale)])])
        return self.op("Mul", [self.op("Div", [xs, den]), self.const(w)])

    def _scale(self, key):
        return float(self.quant["scales"][key]) / 32767.0

    def _qdq(self, x, key):
        """float -> uint16 quantize/dequantize pair with calibrated per-tensor scale (fp32 inside)."""
        s = self.const(np.asarray(self._scale(key), np.float32), keep_dtype=True)
        z = self.const(np.asarray(ZP, np.uint16), keep_dtype=True)
        if self.dt == np.float16:
            x = self.op("Cast", [x], to=TP.FLOAT)
        q = self.op("QuantizeLinear", [x, s, z])
        return self.op("DequantizeLinear", [q, s, z])

    def linear(self, x, w, in_key=None, out_key=None, wtag=None):
        """x @ w.T with w [out,in]. In quant mode: QDQ(x) @ DQ(W_int) -> QDQ(out)."""
        if self.capture and in_key:
            self.caps.append((in_key, x))
        if self.quant is None or in_key is None:
            y = self.op("MatMul", [x, self.const(np.ascontiguousarray(w.T))])
        else:
            bits = self.quant.get("bits", {}).get(wtag, self.quant.get("default_bits", 8))
            qmax = 127 if bits == 8 else 7
            wt = np.ascontiguousarray(w.T).astype(np.float32)                 # [in,out]
            sc = np.maximum(np.abs(wt).max(0), 1e-8) / qmax                    # per output channel
            qw = np.clip(np.round(wt / sc), -qmax - 1, qmax).astype(np.int8)
            wn = self.name("qw")
            if bits == 4:
                self.inits.append(h.make_tensor(wn, TP.INT4, qw.shape, qw.flatten().tolist()))
            else:
                self.inits.append(onnx.numpy_helper.from_array(qw, wn))
            sn = self.const(sc.astype(np.float32), keep_dtype=True)
            wd = self.op("DequantizeLinear", [wn, sn], axis=1)
            mk = (in_key, x)
            if mk not in self._qin:
                self._qin[mk] = self._qdq(x, in_key)
            ym = self.op("MatMul", [self._qin[mk], wd])
            y = self._qdq(ym, out_key) if out_key else ym
            if self.dt == np.float16:
                y = self.op("Cast", [y], to=TP.FLOAT16)
        if self.capture and out_key:
            self.caps.append((out_key, y))
        return y

    def silu(self, x):
        return self.op("Mul", [x, self.op("Sigmoid", [x])])

    def rot(self, x, cos, sin):
        x1 = self.op("Slice", [x, self.ints([0]), self.ints([HD // 2]), self.ints([3])])
        x2 = self.op("Slice", [x, self.ints([HD // 2]), self.ints([HD]), self.ints([3])])
        rh = self.op("Concat", [self.op("Neg", [x2]), x1], axis=3)
        return self.op("Add", [self.op("Mul", [x, cos]), self.op("Mul", [rh, sin])])


def build_chunk(st: Safetensors, l0, l1, T, nf, dtype=np.float16, rscale=1.0, name="chunk", quant=None, capture=False):
    """Layers [l0,l1). Inputs: hidden[1,T,D], valid[1,T,1], cos/sin_{full,sl}[1,1,T,128], mask_{full,sl}[1,1,4T,Nc+T],
    csel[KC+T,KC], cstate_i[1,D,6], pk_i/pv_i[1,4,Nc,128]. Outputs: hidden, cstate_i_out, nk_i/nv_i[1,4,T,128]
    (+ cap_* calibration taps when capture=True)."""
    g = G(dtype, quant, capture)
    W = lambda i, k: st.get(f"model.layers.{i}.{k}", np.float32)
    ins = [("hidden", [1, T, D]), ("valid", [1, T, 1])]
    outs = []
    kinds = {LT[i] for i in range(l0, l1)}
    if "full_attention" in kinds:
        ins += [("cos_full", [1, 1, T, HD]), ("sin_full", [1, 1, T, HD]), ("mask_full", [1, 1, NH // NKV * T, nf + T])]
    if "sliding_attention" in kinds:
        ins += [("cos_sl", [1, 1, T, HD]), ("sin_sl", [1, 1, T, HD]), ("mask_sl", [1, 1, NH // NKV * T, WIN + T])]
    if "conv" in kinds:
        ins.append(("csel", [KC + T, KC]))
    x = "hidden"
    for i in range(l0, l1):
        K = lambda t, i=i: f"{i}.{t}"
        res = x
        xn = g.rms(x, W(i, "operator_norm.weight"), rscale)
        if LT[i] == "conv":
            ins.append((f"cstate_{i}", [1, D, KC]))
            wi = W(i, "conv.in_proj.weight")                                     # [3D, D]: B, C, x
            Bv = g.linear(xn, wi[:D], K("inproj.in"), K("B.out"), "inproj")
            Cv = g.linear(xn, wi[D:2 * D], K("inproj.in"), K("C.out"), "inproj")
            Xv = g.linear(xn, wi[2 * D:], K("inproj.in"), K("X.out"), "inproj")
            bx = g.op("Mul", [g.op("Mul", [Bv, Xv]), "valid"])
            seq = g.op("Concat", [f"cstate_{i}", g.op("Transpose", [bx], perm=[0, 2, 1])], axis=2)   # [1,D,6+T]
            # new state = the KC values preceding the first padded row: one-hot selection matrix (host-built) instead
            # of a dynamic slice, because padding sits at the END of the chunk (state must directly precede real tokens)
            outs.append((f"cstate_{i}_out", g.op("MatMul", [seq, "csel"]), [1, D, KC]))
            cw = g.const(W(i, "conv.conv.weight"))                                                    # [D,1,7]
            co = g.op("Conv", [seq, cw], group=D, kernel_shape=[7], pads=[0, 0], strides=[1])         # [1,D,T]
            co = g.op("Transpose", [co], perm=[0, 2, 1])
            o = g.linear(g.op("Mul", [Cv, co]), W(i, "conv.out_proj.weight"), K("outproj.in"), K("outproj.out"), "outproj")
        else:
            kind = "full" if LT[i] == "full_attention" else "sl"
            nc = nf if kind == "full" else WIN
            ins += [(f"pk_{i}", [1, NKV, nc, HD]), (f"pv_{i}", [1, NKV, nc, HD])]
            q = g.op("Reshape", [g.linear(xn, W(i, "self_attn.q_proj.weight"), K("attn.in"), K("q.out"), "q"), g.ints([1, T, NH, HD])])
            k = g.op("Reshape", [g.linear(xn, W(i, "self_attn.k_proj.weight"), K("attn.in"), K("k.out"), "k"), g.ints([1, T, NKV, HD])])
            v = g.op("Reshape", [g.linear(xn, W(i, "self_attn.v_proj.weight"), K("attn.in"), K("v.out"), "v"), g.ints([1, T, NKV, HD])])
            q = g.rms(q, W(i, "self_attn.q_norm.weight") * (HD ** -0.5), rscale)   # attention scale folded into q_norm
            k = g.rms(k, W(i, "self_attn.k_norm.weight"), rscale)
            tr = lambda t: g.op("Transpose", [t], perm=[0, 2, 1, 3])
            q, k, v = tr(q), tr(k), tr(v)
            q = g.rot(q, f"cos_{kind}", f"sin_{kind}")
            k = g.rot(k, f"cos_{kind}", f"sin_{kind}")
            outs += [(f"nk_{i}", k, [1, NKV, T, HD]), (f"nv_{i}", v, [1, NKV, T, HD])]
            Kc = g.op("Concat", [f"pk_{i}", k], axis=2)
            Vc = g.op("Concat", [f"pv_{i}", v], axis=2)
            qg = g.op("Reshape", [q, g.ints([1, NKV, NH // NKV * T, HD])])
            s = g.op("Add", [g.op("MatMul", [qg, g.op("Transpose", [Kc], perm=[0, 1, 3, 2])]), f"mask_{kind}"])
            p = g.op("Softmax", [s], axis=-1)
            c = g.op("Reshape", [g.op("MatMul", [p, Vc]), g.ints([1, NH, T, HD])])
            gate = g.op("Sigmoid", [g.linear(xn, W(i, "self_attn.gate_proj.weight"), K("attn.in"), K("gate.out"), "gate")])   # [1,T,16]
            gate = g.op("Unsqueeze", [g.op("Transpose", [gate], perm=[0, 2, 1]), g.ints([3])])
            c = g.op("Reshape", [tr(g.op("Mul", [c, gate])), g.ints([1, T, NH * HD])])
            o = g.linear(c, W(i, "self_attn.o_proj.weight"), K("o.in"), K("o.out"), "o")
        x = g.op("Add", [res, g.rms(o, W(i, "post_operator_norm.weight"), rscale)])
        res = x
        y = g.rms(x, W(i, "ffn_norm.weight"), rscale)
        a = g.silu(g.linear(y, W(i, "feed_forward.w1.weight"), K("ffn.in"), K("w1.out"), "w1"))
        b = g.linear(y, W(i, "feed_forward.w3.weight"), K("ffn.in"), K("w3.out"), "w3")
        f = g.linear(g.op("Mul", [a, b]), W(i, "feed_forward.w2.weight"), K("w2.in"), K("w2.out"), "w2")
        x = g.op("Add", [res, g.rms(f, W(i, "post_ffn_norm.weight"), rscale)])
    outs.insert(0, ("hidden_out", x, [1, T, D]))
    for key, t in g.caps:
        outs.append((f"cap_{key}", t, None))
    return finish(g, ins, outs, name)


def finish(g, ins, outs, name):
    vi = lambda n, s: h.make_tensor_value_info(n, g.tp, s)
    out_vi, ident, seen = [], [], set()
    for n, src, s in outs:
        if n in seen:
            continue
        seen.add(n)
        ident.append(h.make_node("Identity", [src], [n]))
        out_vi.append(vi(n, s))
    graph = h.make_graph(g.nodes + ident, name, [vi(n, s) for n, s in ins], out_vi, g.inits)
    m = h.make_model(graph, opset_imports=[h.make_opsetid("", 21)])
    m.ir_version = 10
    return m
