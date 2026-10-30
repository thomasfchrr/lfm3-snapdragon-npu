#!/bin/sh
# Rebuilds every artifact from scratch, in the order used for the published results.
# Windows ARM64 + Python 3.12 ARM64 venv in .venv (see README). Takes roughly an hour and ~25 GB of disk.
set -e
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=$(pwd -W 2>/dev/null || pwd)

# 1. original weights (LFM Open License v1.0, see NOTICE.md)
$PY -c "from huggingface_hub import snapshot_download as s; s('LiquidAI/LFM3-1.7B', local_dir='model_hf', allow_patterns=['*.json','*.py','*.jinja','*.safetensors','LICENSE','README.md'])"

# 2. calibration of activation ranges (fp32 capture graphs on CPU)
$PY scripts/10_calibrate.py

# 3. final NPU graphs: A16W8, decode (T=1) + prefill (T=64)
$PY scripts/11_quant_export.py w8b amax 1.1 8 1,64 fp32 8

# 4. comparison graphs: NPU fp16 (single size, T=8) and CPU fp32 baseline (decode T=1)
$PY scripts/02_export.py fp16 6 8 "$R\\onnx_T8"
$PY scripts/02_export.py fp32 6 1 "$R\\onnx_fp32_T1"
$PY scripts/02_export.py fp32 6 32 "$R\\onnx_fp32_T32"

# 5. fp32 reference for the quality metrics, then everything else
$PY scripts/12_eval_quality.py ref cpu fp32 "$R\\onnx_fp32_T1"
sh scripts/run_all_bench.sh
sh scripts/run_agent_bench.sh
$PY scripts/07_verify_npu.py "$R\\onnx_w8b" w8b
