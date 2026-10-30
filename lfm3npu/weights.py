"""Minimal torch-free safetensors reader (BF16 -> fp32/fp16 via numpy)."""
import json, struct
import numpy as np

_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I64": np.int64, "I32": np.int32}


class Safetensors:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.meta = json.loads(f.read(n))
        self.meta.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self):
        return list(self.meta)

    def shape(self, k):
        return tuple(self.meta[k]["shape"])

    def get(self, k, dtype=np.float32):
        m = self.meta[k]
        a, b = m["data_offsets"]
        raw = np.frombuffer(self.mm[self.base + a:self.base + b], dtype=_DT[m["dtype"]]).reshape(m["shape"])
        if m["dtype"] == "BF16":
            raw = (raw.astype(np.uint32) << 16).view(np.float32)
        return raw.astype(dtype, copy=False)
