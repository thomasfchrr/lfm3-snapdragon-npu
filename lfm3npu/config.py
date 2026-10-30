import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "model_hf")
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    CFG = json.load(f)
D = CFG["hidden_size"]; FF = CFG["intermediate_size"]; NL = CFG["num_hidden_layers"]
NH = CFG["num_attention_heads"]; NKV = CFG["num_key_value_heads"]; HD = CFG["head_dim"]
EPS = CFG["norm_eps"]; KC = CFG["conv_L_cache"] - 1   # conv state length (6)
WIN = CFG["sliding_window"]; VOCAB = CFG["vocab_size"]; LT = CFG["layer_types"]


def rope_tables(positions, kind):
    """cos/sin [T,128] in fp32. 'full': theta 1e6, only first 32 angles rotate (zero tail); 'sliding': theta 1e4 full."""
    rp = CFG["rope_parameters"]["full_attention" if kind == "full" else "sliding_attention"]
    pf = rp.get("partial_rotary_factor", 1.0)
    ang = int(pf * HD) // 2
    inv = 1.0 / (rp["rope_theta"] ** (np.arange(0, 2 * ang, 2, dtype=np.float64) / HD))
    inv = np.concatenate([inv, np.zeros(HD // 2 - ang)])
    f = np.asarray(positions, dtype=np.float64)[:, None] * inv[None]
    e = np.concatenate([f, f], -1)
    return np.cos(e).astype(np.float32), np.sin(e).astype(np.float32)

import numpy as np  # noqa: E402
