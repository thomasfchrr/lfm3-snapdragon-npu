"""Interactive tool-calling agent running LFM3-1.7B on the Snapdragon X Hexagon NPU.
usage: python scripts/lfm3_cli.py [--think] [--model-dir onnx_w8b] [--tag w8b]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lfm3npu.engine import Engine
from lfm3npu.session import ChatModel
from lfm3npu.agent import run_agent
from lfm3npu.tools import Notes, make_registry

ap = argparse.ArgumentParser()
ap.add_argument("--think", action="store_true", help="enable the model's <think> reasoning (slower, longer)")
ap.add_argument("--model-dir", default="onnx_w8b")
ap.add_argument("--tag", default="w8b")
ap.add_argument("--backend", default="npu", choices=["npu", "cpu"])
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("loading (about a minute: graphs are compiled for the NPU)...")
e = Engine(os.path.join(root, a.model_dir), backend=a.backend, tag=a.tag, log=lambda *x: None)
cm = ChatModel(e)
schemas, fns = make_registry(Notes())
print("tools:", ", ".join(s["function"]["name"] for s in schemas), "- empty line to quit")
while True:
    try:
        q = input("\n> ").strip()
    except EOFError:
        break
    if not q:
        break
    ans, trace, el = run_agent(cm, q, schemas, fns, thinking=a.think, max_new=700 if a.think else 300)
    for s in trace:
        for c, r in zip(s["calls"], s.get("results", [])):
            print(f"  [tool] {c['name']}({c['arguments']}) -> {r}")
    st = trace[-1]["stats"]
    print(ans)
    print(f"  ({el:.1f}s, {len(trace)} step(s), decode {st['decode_tok_s']:.1f} tok/s)")
