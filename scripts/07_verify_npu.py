"""Proof that the graphs run on the NPU: (1) QNN EP node coverage log, (2) strict no-CPU-fallback session,
(3) ORT profiler events grouped by execution provider, for one chunk and the head."""
import sys, json, glob, os, collections, numpy as np, onnxruntime as ort
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lfm3npu.npu import register, npu_devices

d = sys.argv[1]; tag = sys.argv[2]
register()
res = {}
for fn in sorted(glob.glob(os.path.join(d, f"chunk_*_T*_{tag}.onnx"))) + [os.path.join(d, f"head_{tag}.onnx")]:
    so = ort.SessionOptions()
    so.log_severity_level = 1
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(os.path.join(ROOT, r"results"), "ortprof_" + os.path.basename(fn)[:-5])
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    so.add_provider_for_devices(npu_devices(), {"htp_performance_mode": "burst", "enable_htp_fp16_precision": "1"})
    s = ort.InferenceSession(fn, sess_options=so)
    feeds = {i.name: np.zeros(i.shape, np.float16 if "fp16" in tag else np.float32) for i in s.get_inputs()}
    for _ in range(5):
        s.run(None, feeds)
    prof = s.end_profiling()
    ev = json.load(open(prof))
    cnt = collections.Counter()
    dur = collections.defaultdict(float)
    for e in ev:
        if e.get("cat") == "Node":
            p = e["args"].get("provider", "?")
            cnt[p] += 1; dur[p] += e["dur"]
    nodes = len(onnx_nodes) if (onnx_nodes := __import__("onnx").load(fn, load_external_data=False).graph.node) else 0
    res[os.path.basename(fn)] = {"onnx_nodes": nodes, "profile_node_events_by_provider": dict(cnt),
                                 "time_us_by_provider": dict(dur), "session_providers": s.get_providers()}
    print(os.path.basename(fn), json.dumps(res[os.path.basename(fn)], indent=1))
json.dump(res, open(os.path.join(ROOT, r"results\npu_verification_") + tag + ".json", "w"), indent=1)
