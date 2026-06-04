<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=implementations&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=papers%20%3E%20code%20%3E%20experiments%20in%20JAX%20from%20scratch&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

</div>

---

## The Setup

This folder is where the ideas become runnable code.

The goal is to take the papers and concepts I care about and implement them in a way that makes the mechanics visible. No shortcuts, no hidden frameworks, no relying on a library to explain the hard part for me. If the implementation is good, I should be able to read it, test it, break it, and learn from the failure.

This is the part of the repo where the abstractions have to survive contact with actual code.

---

<div align="center">

## What's Being Built

</div>

```
implementations/
│
├── 01_bpe_tokenizer/     byte-pair encoding from first principles
├── 02_transformer/       attention, MLP blocks, masking, and residual streams
├── 03_training_loop/     forward pass, loss, backward pass, optimizer, logs
├── 04_lora/              low-rank adaptation and parameter-efficient tuning
└── 05_chinchilla_laws/   scaling laws, compute allocation, and training tradeoffs
```

---

<div align="center">

## The Principles

</div>

**Implement the paper, not the legend.**  
I try to match the actual method and assumptions, not the simplified version people repeat online.

**Keep the code inspectable.**  
If a function hides too much, I rewrite it until I can reason about the full data flow.

**Prefer small, testable pieces.**  
The fastest way to debug a model is to make the smallest piece fail in a controlled way.

**Stay close to JAX.**  
The point is to understand how the computation works, not to wrap it in a heavier abstraction.

---

<div align="center">

## Current Focus

<!-- Update this section as the work changes -->

```
┌─────────────────────────────────────────────────────────┐
│  IMPLEMENTATIONS — ACTIVE                               │
│                                                         │
│  Building: BPE tokenizer + transformer training stack  │
│  Reading:  papers while matching behavior in code     │
│  Goal:     make every component understandable        │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## Why This Folder Exists

This repo is not meant to be a notes archive. It is meant to produce working systems.

The implementations folder is where I check whether I actually understand what I read. A clean implementation tells me the idea was real. A broken one tells me where my understanding is incomplete. Both are useful.

The point is not to reach the most polished abstraction. The point is to make the hard parts visible enough that I can explain them and debug them later.

---

## Folder Guide

### 01_bpe_tokenizer
Build a byte-pair encoding tokenizer from scratch, including token merges, vocabulary construction, and encoding logic.

### 02_transformer
Implement the transformer stack: token embeddings, attention, residual connections, feed-forward layers, and masking.

### 03_training_loop
Build the end-to-end loop: batching, forward pass, loss, backward pass, gradient updates, evaluation, and logging.

### 04_lora
Add low-rank adaptation to a small language model and compare the parameter-efficient path to full fine-tuning.

### 05_chinchilla_laws
Work through the scaling-law side of training: compute budgets, parameter/data tradeoffs, and what the Chinchilla result implies.

---

## Working Style

- Start with a minimal version that runs.
- Add tests or sanity checks before adding complexity.
- Compare outputs against expected shapes, ranges, and invariants.
- Keep the code and notes close together so debugging stays practical.

If the implementation gets hard to read, the design is probably too clever.

---

## Skill Targets

| Skill | Why it matters |
|-------|----------------|
| Tokenization from scratch | Needed to understand preprocessing and vocabulary design |
| Attention implementation | Core transformer mechanics should be transparent |
| Stable training loops | Small bugs here can hide in large runs |
| Low-rank adaptation | Useful for parameter-efficient fine-tuning and model control |
| Scaling-law reasoning | Necessary for compute-aware training decisions |

---

## Next Steps

1. Make each subfolder self-contained with its own notes, tests, and code.
2. Keep the transformer implementation small enough to inspect line by line.
3. Use the training loop folder to connect the pieces into one runnable system.
4. Add a short README inside each implementation folder once the code starts growing.

<sub>
This folder is the bridge between the theory in foundations and the experiments that come later.
</sub>
