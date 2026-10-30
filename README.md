# lfm3-snapdragon-npu

LFM3-1.7B ([LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)) running on the Hexagon NPU of a Snapdragon X Plus
through ONNX Runtime and the QNN execution provider, with a tool-calling agent loop on top.

The ONNX graphs are built by the scripts in this repository, without Qualcomm's conversion tools (QAIRT, AI Hub). Quantization and calibration
are implemented here as well.

Independent project, not affiliated with Liquid AI, Qualcomm or Samsung. The repository contains no weights and no graph derived from the model:
the scripts download the original weights and generate the graphs locally (see [License](#license) and [NOTICE.md](NOTICE.md)).

Measurement setup: Samsung Copilot+ PC, Snapdragon X Plus X1P42100 (8 cores, Hexagon NPU), 15.6 GB RAM, Windows 11 ARM64,
ONNX Runtime 1.31.1, onnxruntime-qnn 2.7.0. A single machine and a single driver version.

## Results

| | CPU fp32 (ORT, 8 cores) | NPU fp16 | NPU A16W8 |
|---|---|---|---|
| Decode (tok/s) | 5.0 to 5.4 | 16.6 | 25.4 (39 ms/token) |
| Prefill (tok/s) | 85 (T=32 graph) | 140 to 147 | about 420 |
| Perplexity (203 tokens) | 5.01 | 5.10 | 5.00 |
| Top-1 agreement with fp32 | n/a | 98.5 % | 97.0 % |
| Agent suite, 12 tasks: correct tools / correct answers | 11 / 12, 12 / 12 | 11 / 12, 12 / 12 | 11 / 12, 12 / 12 |
| Agent suite duration | 325 s | 43 s | 28 s |

With A16W8, decode throughput is 25.4 to 25.6 tok/s for contexts of 32, 512 and 1536 tokens. Compared with CPU fp32, the gain is about 4.8x
for decode and 5x for prefill.

Caveats:

* The CPU agent run uses the T=1 graphs (token-by-token prefill, 6 tok/s). With the T=32 graph (85 tok/s) the duration gap on this suite would be smaller. The decode gap is the directly comparable figure.
* The CPU reference is ONNX Runtime fp32. A quantized engine (llama.cpp Q4, for example) would be faster; it was not measured.
* Power and temperature were not measured.
* Quality is evaluated on 203 tokens (4 passages). The perplexity difference between 5.00 and 5.01 is within measurement noise. The 12 agent tasks are a consistency check, not a benchmark.
* The single "tools" failure is identical on CPU and NPU: for "capital of Japan in one word", the model calls `list_notes` instead of answering directly.

## What runs on the NPU

The transformer compute runs on the NPU: 24 layers (projections, short convolutions, GQA attention with RoPE, softmax, normalizations, MLP) and the
`lm_head` (128,000 x 2048). Verification (`results/npu_verification_w8b.json`):

* sessions are created with `session.disable_cpu_ep_fallback=1`: an unsupported node makes session creation fail;
* the ORT profiler shows every graph (4 blocks of 6 layers in 2 sizes, plus the head) executing as a single `QNNExecutionProvider` node, with no CPU node.

Still on the CPU: the embedding table lookup (one `gather`), computation of the RoPE tables and masks, KV cache updates and sampling.

## Model architecture

Read from the code of the model's repository.

* 24 layers, d = 2048, SwiGLU MLP of 7680, vocabulary of 128,000, embeddings tied to the head.
* Layer order: conv, conv, full, conv, sliding, conv, full, ... i.e. 12 convolution layers, 6 global-attention layers and 6 sliding-window (256) attention layers.
* Convolution layer: `in_proj` (2048 to 3 x 2048, split into B, C, x), `h = B * x`, causal depthwise convolution of width 7, output `out_proj(C * conv(h))`. The state kept between calls is 6 values per channel.
* Attention: 16 heads, 4 KV heads, `head_dim` 128, RMSNorm on Q and K, per-head sigmoid gate on the output. RoPE with `rope_theta` 1e6 and partial rotation (32 of 64 angles) for global attention, full 1e4 for the sliding window.
* RMSNorm before and after every operator and every MLP, final `embedding_norm` after the last layer.

The remote code of the model repository is not executed (PyTorch is not available for Windows ARM64 with Python 3.12). An independent fp32 numpy reference (`lfm3npu/ref_numpy.py`) was written from that code
and serves as ground truth: the fp32 ONNX graphs on CPU match it to within 1e-7.

## Layout

```
lfm3npu/
  weights.py    torch-free safetensors reader (BF16 to fp32/fp16)
  ref_numpy.py  independent fp32 reference
  graphs.py     static ONNX graph builder (float and A16Wx QDQ), calibration taps
  export.py     split into blocks of 6 layers + head, several graph sizes
  engine.py     host runtime: KV and convolution caches, masks, RoPE, session chaining
  session.py    generation with prefix snapshots (system prompt and tools prefilled once)
  chat.py       chat template (jinja2), sampling, tool-call parsing
  agent.py, tools.py   agent loop and test tools (calculator, mock weather, time, notes)
scripts/        numbered pipeline, interactive CLI, benchmark runners
scripts/experiments/   probes and ablations used during development
results/        raw measurements (JSON) and calibration statistics
```

Design choices:

1. Fully static shapes. Caches are explicit inputs and outputs: the graph receives the cache (2048 positions for global attention, 256 for the sliding window) and returns only the new keys and values, which the host inserts. No `ScatterND`, no dynamic shape.
2. Right padding and a selection matrix for the convolution state. With left padding, the convolution state is separated from the real tokens and decoding becomes incoherent while prefill stays correct. Real tokens are therefore placed at the head of the block and the new state is extracted by a `MatMul` with a one-hot matrix supplied by the host.
3. A single graph size for prefill and decode in fp16, because the HTP address space (about 4 GB) is shared by all graphs of a process. With int8 weights, two sizes (T=1 and T=64) coexist.
4. GQA without repeating KV: the 4 query heads of a group are folded into the row dimension (`[1,4,4T,128]`) and the mask is repeated on the host. The factor `1/sqrt(128)` is folded into the `q_norm` weight.
5. A16W8 quantization: uint16 `QuantizeLinear`/`DequantizeLinear` pairs (calibrated per-tensor scale) around every large matmul, per-output-channel int8 weights. Norms, RoPE, softmax and gates stay in floating point. `in_proj` is split into three matmuls (B, C, x) to get one scale per output.
   Calibration on 21 sequences (12,000 tokens: model card, code, multilingual text, tool conversations), running the fp32 graphs with capture taps on the CPU.

## Discarded approaches

* Two fp16 graph sizes at once: at the 6th loaded graph (about 4.4 GB of weights), `QNN_GRAPH_ERROR_INVALID_HANDLE`. Address-space limit of the NPU shared by the process. Workaround: a single size in fp16 (T=8, 16.6 tok/s), then two sizes in int8.
* Weight-only int8 or int4 (`DequantizeLinear` then float `MatMul`): nodes are sent back to the CPU, hence rejected in strict mode. Only a fully QDQ graph (activations included) is accepted.
* Per-channel int4 weights on the MLPs, round-to-nearest without GPTQ or AWQ: accepted by the NPU, but the model collapses (perplexity around 317,000) and decoding does not speed up (23.4 tok/s, same as int8). The bottleneck is no longer weight bandwidth.
* fp16 ablations: attention cache length (256 to 2048) and normalizations have little influence on the time of a block. fp16 decoding is limited by weight reads (about 55 to 60 GB/s), hence the gain from int8 (16.6 to 25.4 tok/s).

## Limitations

* Maximum context of 2048 tokens (size of the global-attention cache in the graph), versus 262,144 for the model. Configurable at export time (`nf`), at the cost of more memory.
* A prefix snapshot is reused only if the new prompt starts with the same tokens. The template strips reasoning from previous turns, which sometimes forces a longer prefill.
* Short calibration and small quality evaluation. No standard benchmark (MMLU, BFCL, etc.) was run. Differences on long tasks are possible.
* The reasoning mode (enabled by default in the template) generates about 5 times more tokens: 86 s versus 28 s on the agent suite. It is disabled by default in the CLI (`--think` to enable it).
* Loading takes about 70 s at every start (graph compilation for the NPU, no context cache) and about 7.8 GB of host RAM.

## Usage

Requirements: Windows ARM64, Python 3.12 ARM64 (`onnxruntime-qnn` packages are not available for emulated x64 Python), about 25 GB of disk for the full pipeline.

Quick start (about 10 min, 7 GB of disk). The calibration statistics are provided in `results/calib_stats.json`.

```bash
py -3.12-arm64 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 1. original weights (LFM license, see NOTICE.md)
.venv\Scripts\python -c "from huggingface_hub import snapshot_download as s; s('LiquidAI/LFM3-1.7B', local_dir='model_hf', allow_patterns=['*.json','*.py','*.jinja','*.safetensors','LICENSE','README.md'])"

# 2. local generation of the A16W8 graphs (decode T=1, prefill T=64) into onnx_w8b/
.venv\Scripts\python scripts\11_quant_export.py w8b amax 1.1 8 1,64 fp32 8

# 3. interactive agent
.venv\Scripts\python scripts\lfm3_cli.py
```

Full pipeline:

| Script | Purpose |
|---|---|
| `00_smoke.py` | minimal NPU test (matmul, strict mode) |
| `01_one_layer.py` | one layer against the numpy reference (CPU or NPU) |
| `02_export.py <fp16\|fp32> 6 <T,...> <folder>` | export of float graphs (NPU fp16, CPU fp32 baseline) |
| `03_first_run.py` | short generation, comparison with the reference logits |
| `06_agent_demo.py` | agent examples |
| `10_calibrate.py` | activation statistics on CPU fp32, writes `results/calib_stats.json` |
| `11_quant_export.py w8b amax 1.1 8 1,64 fp32 8` | A16W8 graphs (`onnx_w8b/`) |
| `12_eval_quality.py` | perplexity and top-1 agreement against the fp32 reference |
| `13_bench.py` | prefill and decode throughput |
| `14_agent_bench.py` | 12-task agent suite |
| `07_verify_npu.py` | verification of NPU execution (ORT profiler) |
| `run_all_bench.sh`, `run_agent_bench.sh` | sequential runners (never in parallel, for clean measurements) |
| `build_all.sh` | all steps in order |

Each step of `build_all.sh` was run separately; the chained script itself was not replayed in full. The `model_hf/` and `onnx_*/` folders (about 16 GB) are excluded from the repository because they are weights or derivatives of weights.

Raw measurements: `results/bench_*.json`, `results/quality.json`, `results/agent_*_think*.json` (answers and tools of each task), `results/npu_verification_*.json`, `results/calib_stats.json`.
Some calibration and agent test strings are multilingual on purpose.

## License

* Code (`lfm3npu/`, `scripts/`): MIT, see [LICENSE](LICENSE).
* LFM3-1.7B model: LFM Open License v1.0 (Liquid AI). It allows redistribution and derivative works under conditions (copy of the license, notice of modifications, attributions) and limits commercial use to entities with annual revenue below US$10M. This repository does not redistribute the weights; the graphs generated locally are derived from them and follow that license. Details in [NOTICE.md](NOTICE.md).
