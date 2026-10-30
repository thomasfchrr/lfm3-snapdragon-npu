"""ONNX Runtime + QNN (Hexagon HTP) session helpers for Windows on Snapdragon X."""
import onnxruntime as ort
import onnxruntime_qnn as q

_registered = False


def register():
    global _registered
    if not _registered:
        for n in q.get_ep_names():
            ort.register_execution_provider_library(n, q.get_library_path())
        _registered = True


def npu_devices():
    register()
    return [d for d in ort.get_ep_devices()
            if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU]


def make_session(model, npu=True, fp16=True, strict=True, extra_opts=None, ctx_cache=None, log_level=3):
    """npu=True -> QNN HTP with CPU fallback disabled when strict (guarantees 100% NPU or an error)."""
    so = ort.SessionOptions()
    so.log_severity_level = log_level
    if npu:
        register()
        dev = npu_devices()
        assert dev, "no QNN NPU device"
        if strict:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        opts = {"htp_performance_mode": "burst",
                "enable_htp_fp16_precision": "1" if fp16 else "0"}
        if extra_opts:
            opts.update(extra_opts)
        if ctx_cache:
            so.add_session_config_entry("ep.context_enable", "1")
            so.add_session_config_entry("ep.context_file_path", ctx_cache)
        so.add_provider_for_devices(dev, opts)
        return ort.InferenceSession(model if isinstance(model, str) else model, sess_options=so)
    so.intra_op_num_threads = 8
    return ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])
