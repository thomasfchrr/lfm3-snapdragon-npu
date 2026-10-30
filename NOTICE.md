# Notice

This is an independent, unofficial project. It is **not affiliated with, endorsed by, or sponsored by Liquid AI, Inc.,
Qualcomm, or Samsung**. "LFM", "Liquid AI", "Snapdragon", "Hexagon" and other names belong to their respective owners
and are used here only to describe what the code works with.

## What this repository contains

Only original code, documentation and measurement results. **It contains no model weights and no graph derived from them.**
All code in `lfm3npu/` and `scripts/` was written for this project (the model's own `modeling_lfm3.py` was read to understand the
architecture, not copied) and is released under the MIT license (see `LICENSE`).

## The model

The weights are `LiquidAI/LFM3-1.7B`, published by Liquid AI under the **LFM Open License v1.0**:
<https://huggingface.co/LiquidAI/LFM3-1.7B> (the license text is the `LICENSE` file of that repository).

Key points of that license, summarized by the author of this repository (not legal advice, read the original):
* redistribution of the weights and derivative works is allowed, provided that recipients get a copy of the license,
  modified files carry a prominent notice that they were changed, and attribution notices are kept;
* commercial use is licensed only for entities with annual revenue **below US$10 million** (affiliates included);
  above that threshold, commercial use is not licensed by this license;
* trademarks are not licensed.

The scripts download the original weights from Hugging Face and generate the quantized ONNX graphs **locally**. Anyone who
redistributes the generated graphs is subject to the conditions above for those graphs.

## Tooling

Development of this repository was assisted by Claude Sonnet 5.5 (Anthropic).
