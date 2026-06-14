# 04 — LoRA: Low-Rank Adaptation

**Status:** Planned / Not yet built

This folder will implement low-rank adaptation from *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al. 2021):

- Rank decomposition of weight updates
- Applying LoRA to attention projections
- Parameter-efficient fine-tuning vs full fine-tuning
- Merge/unmerge for inference

Prerequisites: the transformer implementation in `02` and a working training loop in `03`.
