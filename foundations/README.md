<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=foundations&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=mathematical%20bedrock%20for%20building%20ML%20from%20first%20principles&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

</div>

---

## The Setup

Before the code gets interesting, the math has to be real.

This folder is where I rebuild the pieces that sit underneath the rest of the repo: linear algebra, calculus, and probability. The goal is not to memorize definitions. The goal is to derive the ideas, implement them, and debug them when they show up inside a real model.

If something breaks later in the implementation folders, this is where I come back to check whether the intuition was actually sound.

---

<div align="center">

## What's Being Built

</div>

```
foundations/
│
├── linear_algebra/    vectors, matrices, projections, eigendecomposition, SVD
├── calculus/          derivatives, Jacobians, chain rule, backprop by hand
└── probability/       distributions, likelihood, KL divergence, Bayesian intuition
```

---

<div align="center">

## The Principles

</div>

**Write it down, then derive it.**  
If I cannot derive the result on paper, I do not really understand it yet.

**Implement it, then inspect it.**  
Small numerical examples should match the theory before I trust a bigger system.

**Connect it back to code.**  
Every identity should eventually point at something I will use in JAX.

**Keep the examples small.**  
Tiny matrices and simple distributions reveal mistakes faster than ambitious notation.

---

<div align="center">

## Current Focus

<!-- Update this section as the work changes -->

```
┌─────────────────────────────────────────────────────────┐
│  FOUNDATIONS — ACTIVE                                   │
│                                                         │
│  Building: intuition for gradients and matrix algebra   │
│  Reading:  how the math maps onto transformer code     │
│  Goal:     make the rest of the repo easier to debug    │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## Why It Matters

The point is to make these ideas operational, not just familiar.

- Linear algebra explains how attention and embeddings move information around.
- Calculus explains how gradients flow through a computation graph.
- Probability explains loss functions, uncertainty, and optimization objectives.

---

## Working Style

- Short derivations before long implementations.
- Small examples before full tensor shapes.
- Notes on what confused me, not just what worked.
- Connections back to the code I will write later.

---

## Next Steps

1. Build a clean linear algebra refresher.
2. Work through backprop by hand until it feels mechanical.
3. Keep enough probability theory in view to make loss functions and objectives feel natural.

Once those are solid, the implementation folders should move faster and break more cleanly.

<sub>
Part of the same build log as the root project. The difference is that this folder starts with the math instead of the model.
</sub>
