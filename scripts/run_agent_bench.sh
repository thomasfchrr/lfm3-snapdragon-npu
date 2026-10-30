#!/bin/sh
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
R=$(pwd -W 2>/dev/null || pwd)
run() { echo "=== $*"; $PY "$@" 2>&1 | grep -av "^\s*$" | tail -16; }
run scripts/14_agent_bench.py npu_w8a16 npu w8b "$R\\onnx_w8b" 0
run scripts/14_agent_bench.py npu_w8a16 npu w8b "$R\\onnx_w8b" 1
run scripts/14_agent_bench.py npu_fp16 npu fp16 "$R\\onnx_T8" 0
run scripts/14_agent_bench.py cpu_fp32 cpu fp32 "$R\\onnx_fp32_T1" 0
echo AGENT_DONE
