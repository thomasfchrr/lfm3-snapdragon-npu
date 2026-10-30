"""Independent fp32 numpy reference of LFM3 (written from modeling_lfm3.py), layer-streamed to save RAM."""
import numpy as np
from .config import *
from .weights import Safetensors


def rms(x, w):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * w


def silu(x):
    return x / (1 + np.exp(-x))


def rot_half(x):
    return np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)


class RefModel:
    def __init__(self, st=None):
        self.st = st or Safetensors(f"{MODEL_DIR}/model.safetensors")
        self.emb = None

    def W(self, k):
        return self.st.get("model." + k)

    def layer(self, i, h, pos, state):
        """h [T,D] fp32; state dict per layer (conv: last 6 of B*x; attn: k,v lists). Returns h."""
        W = lambda k: self.W(f"layers.{i}.{k}")
        T = h.shape[0]
        x = rms(h, W("operator_norm.weight"))
        if LT[i] == "conv":
            bcx = x @ W("conv.in_proj.weight").T
            B, C, xx = bcx[:, :D], bcx[:, D:2 * D], bcx[:, 2 * D:]
            bx = B * xx
            prev = state.get(i, np.zeros((KC, D), np.float32))
            seq = np.concatenate([prev, bx], 0)                 # [6+T, D]
            w = W("conv.conv.weight")[:, 0, :]                  # [D,7]
            out = np.zeros((T, D), np.float32)
            for k in range(7):
                out += seq[k:k + T] * w[:, k][None]
            state[i] = seq[-KC:]
            o = (C * out) @ W("conv.out_proj.weight").T
        else:
            kind = "full" if LT[i] == "full_attention" else "sliding"
            q = (x @ W("self_attn.q_proj.weight").T).reshape(T, NH, HD)
            k = (x @ W("self_attn.k_proj.weight").T).reshape(T, NKV, HD)
            v = (x @ W("self_attn.v_proj.weight").T).reshape(T, NKV, HD)
            q = rms(q, W("self_attn.q_norm.weight")); k = rms(k, W("self_attn.k_norm.weight"))
            cos, sin = rope_tables(pos, kind)
            q = q * cos[:, None] + rot_half(q) * sin[:, None]
            k = k * cos[:, None] + rot_half(k) * sin[:, None]
            pk, pv, ppos = state.get(i, (np.zeros((0, NKV, HD), np.float32), np.zeros((0, NKV, HD), np.float32), np.zeros(0, int)))
            K = np.concatenate([pk, k], 0); V = np.concatenate([pv, v], 0); P = np.concatenate([ppos, pos])
            state[i] = (K, V, P)
            Kr = np.repeat(K, NH // NKV, 1); Vr = np.repeat(V, NH // NKV, 1)
            s = np.einsum("thd,shd->hts", q, Kr) / np.sqrt(HD)
            m = P[None, :] <= np.asarray(pos)[:, None]
            if kind == "sliding":
                m &= P[None, :] > np.asarray(pos)[:, None] - WIN
            s = np.where(m[None], s, -1e30)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", s, Vr)
            g = 1 / (1 + np.exp(-(x @ W("self_attn.gate_proj.weight").T)))
            a = (a * g[:, :, None]).reshape(T, NH * HD)
            o = a @ W("self_attn.o_proj.weight").T
        h = h + rms(o, W("post_operator_norm.weight"))
        y = rms(h, W("ffn_norm.weight"))
        f = (silu(y @ W("feed_forward.w1.weight").T) * (y @ W("feed_forward.w3.weight").T)) @ W("feed_forward.w2.weight").T
        return h + rms(f, W("post_ffn_norm.weight"))

    def forward(self, ids, pos0=0, state=None, n_layers=NL, return_hidden=False):
        state = {} if state is None else state
        E = self.st.get("model.embed_tokens.weight", np.float32) if self.emb is None else self.emb
        h = E[np.asarray(ids)]
        pos = np.arange(pos0, pos0 + len(ids))
        for i in range(n_layers):
            h = self.layer(i, h, pos, state)
        h = rms(h, self.W("embedding_norm.weight"))
        if return_hidden:
            return h, state
        return h[-1] @ E.T, state
