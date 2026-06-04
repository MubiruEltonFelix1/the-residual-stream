<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=01_bpe_tokenizer&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=build%20byte-pair%20encoding%20from%20scratch%20in%20JAX&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<!-- BADGES -->
![Status](https://img.shields.io/badge/status-active_build-22c55e?style=for-the-badge&labelColor=0f0c29)
![Track](https://img.shields.io/badge/track-tokenization-f59e0b?style=for-the-badge&labelColor=0f0c29)
![Stack](https://img.shields.io/badge/stack-JAX_%2B_NumPy-a78bfa?style=for-the-badge&labelColor=0f0c29)

</div>

---

## The Setup

This folder is where I build a Byte Pair Encoding (BPE) tokenizer from zero.

If you are sixteen and new to this, think of BPE like this: a tokenizer is a machine that cuts text into useful pieces. If you cut too much, everything becomes tiny and slow. If you cut too little, rare words become hard to handle. BPE learns good pieces by repeatedly merging the most common neighboring symbols.

The goal here is not to import a tokenizer and move on. The goal is to understand every step enough to explain it, test it, and debug it.

---

<div align="center">

## What's Being Built

</div>

```
01_bpe_tokenizer/
│
├── README.md                 this guide and workflow
├── bpe_tokenizer.py          tokenizer implementation (to be created)
├── toy_corpus.txt            tiny text sample for fast tests (to be created)
├── tests/                    minimal correctness tests (to be created)
└── notes.md                  implementation notes and mistakes (to be created)
```

---

<div align="center">

## Source Material (What This Refers To)

</div>

When this workflow says "read the source material first," it means:

1. The original BPE compression idea:
   - Gage, 1994, "A New Algorithm for Data Compression"
   - This is where the merge idea comes from.

2. BPE for neural text tokenization:
   - Sennrich, Haddow, and Birch, 2016, "Neural Machine Translation of Rare Words with Subword Units"
   - This is the key paper for NLP-style subword tokenization.

3. Optional practical references (after the papers):
   - GPT-2/modern tokenizer behavior notes (special tokens, byte handling, edge cases)

If you skip this reading, you can still write code, but you will not know whether your code matches the intended algorithm.

---

<div align="center">

## The Workflow (Explained Like You're 16)

</div>

### 1) Read the source material and write the paper note first

Why first: reading gives you the rules of the game.

What to do:
- Create one note in [paper_notes](../../paper_notes).
- Write the claim in your own words.
- Write the merge process in plain language.
- List what you expect to be hard in implementation.

Simple idea: before building a bicycle, learn which part is the brake.

### 2) Define the minimum tokenizer behavior

Why: if "done" is unclear, you will keep changing the target.

Minimum definition for this project:
- train merges from a corpus
- encode text into token ids
- decode token ids back to text
- unknown cases handled in a predictable way

Simple idea: decide what "a working prototype" means before coding.

### 3) Implement a tiny version on a toy corpus

Why: tiny systems fail faster and teach faster.

What tiny means:
- use a corpus of maybe 5-20 short lines
- train a small number of merges
- print merge steps so you can inspect them

Simple idea: first make it work on a small sandbox before a real city.

### 4) Test on a very small dataset before scaling

Why: scaling hides bugs.

What to test:
- encode-decode round trip should preserve text policy
- frequent pairs should merge as expected
- vocabulary size should grow exactly as planned
- deterministic behavior with fixed settings

Simple idea: if a calculator fails on 2 + 2, do not trust it on calculus.

### 5) Record failures and surprises in weekly builds

Why: this is where real learning happens.

What to record in [weekly_builds](../../weekly_builds):
- what failed
- what you thought was true but was wrong
- what fixed it
- what remains confusing

Simple idea: mistakes are not noise; they are the map.

### 6) Only then extend to a full tokenizer

After the tiny version is stable, then add:
- cleaner API
- better speed
- larger corpora
- special tokens and improved text handling

Simple idea: grow from stable foundations, not from guesswork.

---

<div align="center">

## Practical File Workflow

</div>

Use this order every time:

1. Write paper note in [paper_notes](../../paper_notes) first.
2. Add tiny corpus in this folder.
3. Build minimal tokenizer code.
4. Add quick tests.
5. Log weekly progress in [weekly_builds](../../weekly_builds).
6. Refactor only after tests pass.

---

<div align="center">

## Current Focus

<!-- Update this section as work changes -->

```
┌─────────────────────────────────────────────────────────┐
│  01_BPE_TOKENIZER — ACTIVE                              │
│                                                         │
│  Building: tiny BPE train/encode/decode loop           │
│  Reading:  Gage 1994 + Sennrich et al. 2016            │
│  Goal:     correctness first, scale second             │
└─────────────────────────────────────────────────────────┘
```

</div>

---

<div align="center">

## Definition Of Done (Phase 1)

</div>

Phase 1 is complete when:

- I can train merges on a toy corpus.
- I can encode and decode sample text.
- I have tests for round-trip behavior.
- I have one paper note and one weekly build entry.
- I can explain each step without reading code comments.

---

<div align="center">

## Common Traps

</div>

- Merging pairs inconsistently between training and encoding
- Non-deterministic tie-breaking for equally frequent pairs
- Decode not being the true inverse of encode policy
- Jumping to large corpus before proving correctness
- Confusing "fast" with "correct"

---

<div align="center">

## Next Steps

</div>

1. Add the first paper note for BPE in [paper_notes](../../paper_notes).
2. Create a tiny corpus file in this folder.
3. Implement a plain, readable baseline tokenizer.
4. Add 3-5 core tests.
5. Write the first weekly build summary in [weekly_builds](../../weekly_builds).

<sub>
This folder should read like a build journal and a blueprint at the same time.
</sub>
