import sys, time
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.engine import Engine
from lfm3npu.session import ChatModel
from lfm3npu.agent import run_agent
from lfm3npu.tools import Notes, make_registry

d = sys.argv[1]; backend = sys.argv[2] if len(sys.argv) > 2 else "npu"
e = Engine(d, backend=backend, tag="fp16" if backend == "npu" else "fp32", log=lambda *a: None)
cm = ChatModel(e)
schemas, fns = make_registry(Notes())
for q, th in [("What is 127*49 + 13?", False), ("What's the weather in Tokyo and in Paris? Which one is warmer?", False),
              ("Combien font 15% de 240 ?", True)]:
    t = time.time()
    ans, trace, el = run_agent(cm, q, schemas, fns, thinking=th, max_new=500, verbose=True)
    print(f"Q: {q} (thinking={th})\nA: {ans}\n   {el:.1f}s, {len(trace)} steps\n")
