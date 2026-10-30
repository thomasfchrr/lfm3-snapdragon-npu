"""Chat-level wrapper around Engine: prompt-prefix snapshots (system prompt/tool schemas are prefilled once),
streaming generation with sampling, per-phase timing."""
import copy, time
import numpy as np
from . import chat
from .engine import Engine, Cache


def clone(c: Cache) -> Cache:
    return copy.deepcopy(c)


class ChatModel:
    def __init__(self, engine: Engine, seed=0):
        self.e = engine
        self.rng = np.random.default_rng(seed)
        self.snaps = []                     # list of (ids, Cache) snapshots usable as prefixes
        self.stats = []

    def _best_prefix(self, ids):
        best = (0, None)
        for sid, sc in self.snaps:
            n = min(len(sid), len(ids))
            k = 0
            while k < n and sid[k] == ids[k]:
                k += 1
            if k == len(sid) and k > best[0]:      # snapshot is a full prefix of the new prompt
                best = (k, sc)
        return best

    def generate(self, prompt_ids, max_new=256, temperature=0.1, top_k=50, top_p=1.0, rep_penalty=1.1,
                 stream=None, stop_ids=(chat.EOS,), snapshot_at=None, use_prefix=True):
        """snapshot_at: list of token counts (strict prefixes of prompt_ids) to snapshot while prefilling."""
        e = self.e
        t_start = time.perf_counter()
        k, sc = self._best_prefix(prompt_ids) if use_prefix else (0, None)
        if k >= len(prompt_ids):               # need logits of the last prompt token: replay it from the prefix
            k, sc = 0, None
            for sid, s_ in self.snaps:
                if len(sid) < len(prompt_ids) and prompt_ids[:len(sid)] == list(sid) and len(sid) > k:
                    k, sc = len(sid), s_
        cache = clone(sc) if sc is not None else Cache(e.nf, e.dt)
        if len(prompt_ids) + max_new > e.nf:
            max_new = max(1, e.nf - len(prompt_ids))
        todo = prompt_ids[k:]
        marks = sorted(m for m in (snapshot_at or []) if k < m < len(prompt_ids))
        pos = k; logits = None
        for m in marks + [len(prompt_ids)]:
            if m > pos:
                logits = e.prefill(cache, prompt_ids[pos:m]); pos = m
            if m != len(prompt_ids) and m in marks:
                self.snaps.append((list(prompt_ids[:m]), clone(cache)))
        t_pre = time.perf_counter()
        out, hist = [], list(prompt_ids)
        while len(out) < max_new:
            nx = chat.sample(logits, temperature, top_k, top_p, rep_penalty, hist, self.rng)
            if nx in stop_ids:
                break
            out.append(nx); hist.append(nx)
            if stream:
                stream(nx)
            if len(out) >= max_new:
                break
            logits = e.decode(cache, nx)
        t_end = time.perf_counter()
        st = {"prompt_tokens": len(prompt_ids), "prefilled": len(todo), "reused": k, "new_tokens": len(out),
              "prefill_s": t_pre - t_start, "decode_s": t_end - t_pre,
              "prefill_tok_s": len(todo) / max(t_pre - t_start, 1e-9),
              "decode_tok_s": max(len(out) - 1, 0) / max(t_end - t_pre, 1e-9)}
        self.stats.append(st)
        return out, st
