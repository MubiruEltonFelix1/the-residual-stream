<div align="center">

<!-- HEADER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=180&section=header&text=Foundations&fontSize=44&fontColor=ffffff&fontAlignY=38&desc=mathematical%20bedrock%20for%20building%20ML%20from%20first%20principles&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

</div>

---

## The Setup

Before I can build transformers, optimizers, or training loops, I need the math to stop being ceremonial.

This folder is where I work through the pieces that sit underneath the rest of the repo: linear algebra, calculus, and probability. The goal is not to memorize definitions. The goal is to be able to derive, implement, and debug the ideas when they show up inside a real model.

If something breaks later in `implementations/`, this is where I come back to check whether the intuition was actually sound.

---

<div align="center">

## What Lives Here

</div>

```
foundations/
│
├── linear_algebra/    vectors, matrices, projections, eigendecomposition, SVD
├── calculus/          derivatives, Jacobians, chain rule, backprop by hand
└── probability/       distributions, likelihood, KL divergence, Bayesian intuition
```

---

## Why This Exists

I do not want to treat the math as decoration around machine learning.

When I understand the foundations properly, the rest of the repo becomes easier to reason about:

- Linear algebra explains how attention and embeddings move information around.
- Calculus explains how gradients actually flow through a computation graph.
- Probability explains loss functions, uncertainty, and why optimization objectives look the way they do.

The point is to make these ideas operational, not just familiar.

---

<div align="center">

## Working Rule

</div>

**Write it down, then derive it.**  
If I cannot derive the result on paper, I do not really understand it yet.

**Implement it, then inspect it.**  
Small numerical examples should match the theory before I trust a bigger system.

**Connect it back to code.**  
Every theorem or identity should eventually point at something I will use in JAX.

**Keep the examples small.**  
Tiny matrices and simple distributions reveal mistakes faster than ambitious notation.

---

<div align="center">

## What I Want To Be Able To Do

</div>

| Skill | Why it matters |
|-------|----------------|
| Multiply, reshape, and decompose matrices cleanly | Most model internals are matrix operations in disguise |
| Derive gradients by hand | Backprop only makes sense when the chain rule is concrete |
| Recognize stable vs unstable numerical formulas | Training fails for numerical reasons before it fails conceptually |
| Reason about probabilities and expectations | Losses, sampling, and evaluation all depend on this |
| Translate math into JAX code | The real test is whether the derivation survives implementation |

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

## How I Use This Folder

I keep the work in this folder close to the reasoning:

- Short derivations before long implementations.
- Small examples before full tensor shapes.
- Notes on what confused me, not just what worked.
- Connections back to the code I will write later.

If the math stays abstract, it does not stay useful.

---

## Next Steps

The immediate goal is simple:

1. Build a clean linear algebra refresher.
2. Work through backprop by hand until it feels mechanical.
3. Keep enough probability theory in view to make loss functions and objectives feel natural.

Once those are solid, the implementation folders should move faster and break more cleanly.

---

<sub>
Part of the same build log as the root project. The difference is that this folder starts with the math instead of the model.
</sub>
