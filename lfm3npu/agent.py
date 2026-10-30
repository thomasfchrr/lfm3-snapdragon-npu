"""Minimal tool-calling agent loop on top of ChatModel."""
import time
from . import chat

DEFAULT_SYSTEM = "You are a helpful assistant with access to tools. Use a tool whenever it helps."


def run_agent(cm, user, schemas, fns, system=DEFAULT_SYSTEM, max_steps=6, max_new=400, thinking=True,
              temperature=0.1, verbose=False, history=None):
    msgs = history if history is not None else [{"role": "system", "content": system}]
    msgs.append({"role": "user", "content": user})
    trace, t0 = [], time.perf_counter()
    for step in range(max_steps):
        kw = {} if thinking else {"enable_thinking": False}
        prompt = chat.render(msgs, schemas, **kw)
        ids = chat.encode(prompt)
        sys_end = prompt.find("<|im_start|>user")
        snap = [len(chat.encode(prompt[:sys_end]))] if sys_end > 0 else []
        out, st = cm.generate(ids, max_new=max_new, temperature=temperature, snapshot_at=snap + [len(ids) - 1])
        text = chat.decode(out, skip_special=False)
        calls = chat.parse_tool_calls(text)
        content = text.split("</think>")[-1] if "</think>" in text else text
        content = content.split("<tool_call>")[0].strip()
        trace.append({"step": step, "text": text, "calls": calls, "stats": st})
        if verbose:
            print(f"[step {step}] {st['new_tokens']} tok, {st['decode_tok_s']:.1f} tok/s, reused {st['reused']}/{st['prompt_tokens']}\n  {text[-300:]!r}")
        if not calls:
            msgs.append({"role": "assistant", "content": content})
            return content, trace, time.perf_counter() - t0
        msgs.append({"role": "assistant", "content": content, "tool_calls": [
            {"id": str(i), "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for i, c in enumerate(calls)]})
        for i, c in enumerate(calls):
            try:
                res = fns[c["name"]](**c["arguments"])
            except Exception as ex:
                res = f"error: {ex}"
            trace[-1].setdefault("results", []).append(res)
            msgs.append({"role": "tool", "tool_call_id": str(i), "content": str(res)})
    return "(max steps reached)", trace, time.perf_counter() - t0
