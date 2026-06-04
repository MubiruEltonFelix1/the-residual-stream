<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=linear%20algebra&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=vectors%2C%20matrices%2C%20and%20the%20geometry%20behind%20model%20code&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

</div>

---

## The Setup

This is where I rebuild the math that sits under embeddings, attention, and matrix-heavy model code.

The goal is not to collect formulas. The goal is to understand what matrices are doing well enough to derive them, implement them, and debug them when they show up inside a real model.

---

<div align="center">

## What's Being Built

</div>

```
linear_algebra/
│
├── vectors and matrices
├── projections and subspaces
├── eigenvalues and eigenvectors
├── singular value decomposition
└── rank, null space, conditioning, and tensor intuition
```

---

<div align="center">

## The Principles

</div>

**Start small.**  
Tiny matrices expose mistakes faster than full-scale notation.

**Derive it by hand.**  
If I cannot explain the algebra step by step, I do not trust the result yet.

**Check the code path.**  
Every identity should map back to a JAX implementation.

**Keep the geometry visible.**  
If the linear algebra loses its spatial meaning, I am probably skipping the hard part.

---

<div align="center">

## Current Focus

<!-- Update this section as the work changes -->

```
┌─────────────────────────────────────────────────────────┐
│  LINEAR ALGEBRA — ACTIVE                                │
│                                                         │
│  Building: matrix intuition and basis changes          │
│  Reading:  how attention and embeddings use geometry   │
│  Goal:     make tensor code easier to reason about     │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## Why It Matters

If I understand linear algebra properly, I can read model code with less guesswork.

- Attention is mostly matrix multiplication with structure.
- Embeddings are learned coordinate systems.
- Low-rank methods like LoRA only make sense if rank is intuitive.
- Numerical stability often comes down to matrix behavior.

---

## Working Style

- Write a small example first.
- Derive the result by hand.
- Check it against code.
- Keep the notation simple enough that I can explain it back later.

---

## Next Step

Build a clean set of notes and small examples that map directly to the transformer code in the rest of the repo.