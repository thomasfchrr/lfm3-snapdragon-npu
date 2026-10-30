"""Agentic task suite (tool calling, multi-step, no-tool, formatting). usage: 14_agent_bench.py NAME BACKEND TAG ONNX_DIR [think=0|1]"""
import sys, json, time, re, numpy as np
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R = ROOT
name, backend, tag, d = sys.argv[1:5]
think = (sys.argv[5] == "1") if len(sys.argv) > 5 else False
from lfm3npu.engine import Engine
from lfm3npu.session import ChatModel
from lfm3npu.agent import run_agent
from lfm3npu.tools import Notes, make_registry

e = Engine(d, backend=backend, tag=tag, log=lambda *a: None)
cm = ChatModel(e, seed=0)

# (id, prompt, expected tool names (set, order-free), predicate on final answer)
TASKS = [
    ("calc_simple", "What is 127*49 + 13?", {"calculator"}, lambda a: "6236" in a.replace(",", "").replace(" ", "")),
    ("calc_fr", "Combien font 15% de 240 ?", {"calculator"}, lambda a: "36" in a),
    ("calc_pow", "Compute 2**16 using the calculator.", {"calculator"}, lambda a: "65536" in a.replace(",", "").replace(" ", "")),
    ("weather_one", "What's the weather in Tokyo?", {"get_weather"}, lambda a: "22" in a),
    ("weather_two", "Compare the weather in Paris and London. Which city is warmer?", {"get_weather"}, lambda a: "paris" in a.lower() and "18" in a),
    ("time", "What time is it right now?", {"get_time"}, lambda a: "14" in a),
    ("note_save", "Save a note titled 'shopping' with content 'milk, eggs, bread'.", {"save_note"}, lambda a: "shopping" in a.lower()),
    ("note_roundtrip", "Save a note called 'idea' containing 'build an NPU agent', then read it back to me.", {"save_note", "read_note"}, lambda a: "npu" in a.lower()),
    ("chain_weather_calc", "Get the weather in Lyon, then multiply its temperature in Celsius by 3 with the calculator.", {"get_weather", "calculator"}, lambda a: "51" in a),
    ("no_tool_hello", "Say hello in French in one short sentence.", set(), lambda a: any(w in a.lower() for w in ("bonjour", "salut", "coucou"))),
    ("no_tool_fact", "What is the capital of Japan? Answer in one word.", set(), lambda a: "tokyo" in a.lower()),
    ("json_format", "Reply ONLY with a JSON object with keys name and age for a person called Alice aged 30.", set(),
     lambda a: bool(re.search(r'"name"\s*:\s*"Alice"', a)) and bool(re.search(r'"age"\s*:\s*30', a))),
]
results = []
for tid, q, exp_tools, check in TASKS:
    schemas, fns = make_registry(Notes())
    t0 = time.time()
    ans, trace, el = run_agent(cm, q, schemas, fns, thinking=think, max_new=700 if think else 300, max_steps=6)
    used = [c["name"] for s in trace for c in s["calls"]]
    toks = sum(s["stats"]["new_tokens"] for s in trace)
    dec = sum(s["stats"]["decode_s"] for s in trace)
    # no-tool tasks must call nothing; tool tasks must call at least the expected tools
    ok_tools = (len(used) == 0) if not exp_tools else bool(exp_tools <= set(used))
    ok_ans = bool(check(ans))
    results.append({"id": tid, "tools_used": used, "tools_ok": ok_tools, "answer_ok": ok_ans, "answer": ans[:200], "steps": len(trace),
                    "gen_tokens": toks, "wall_s": el, "decode_tok_s": toks / max(dec, 1e-9)})
    print(f"{tid:20s} tools={used} tools_ok={ok_tools} ans_ok={ok_ans} steps={len(trace)} toks={toks} {el:.1f}s", flush=True)
summ = {"name": name, "thinking": think, "n": len(results), "tools_ok": sum(r["tools_ok"] for r in results),
        "answer_ok": sum(r["answer_ok"] for r in results), "wall_s": sum(r["wall_s"] for r in results),
        "gen_tokens": sum(r["gen_tokens"] for r in results)}
print(json.dumps(summ))
json.dump({"summary": summ, "tasks": results}, open(R + rf"\results\agent_{name}_think{int(think)}.json", "w"), indent=1)
