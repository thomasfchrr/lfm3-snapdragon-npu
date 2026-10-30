#!/bin/sh
# Sequential (never concurrent) so timings are clean.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=$(pwd -W 2>/dev/null || pwd)
run() { echo "=== $*"; $PY "$@" 2>&1 | grep -av "^\s*$" | tail -25; }
run scripts/13_bench.py npu_w8a16 npu w8b "$R\\onnx_w8b"
run scripts/13_bench.py npu_fp16 npu fp16 "$R\\onnx_T8"
run scripts/13_bench.py cpu_fp32_T1 cpu fp32 "$R\\onnx_fp32_T1"
run scripts/13_bench.py cpu_fp32_T32 cpu fp32 "$R\\onnx_fp32_T32"
run scripts/12_eval_quality.py npu_w8b npu w8b "$R\\onnx_w8b"
echo ALL_DONE
