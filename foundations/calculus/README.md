<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=calculus&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=derivatives%2C%20gradients%2C%20and%20the%20chain%20rule%20in%20practice&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

</div>

---

## The Setup

This is where I work through derivatives, gradients, and the chain rule until they feel mechanical instead of magical.

The goal is not formal elegance. The goal is to understand how gradients move through a computation graph so I can implement backpropagation and debug training behavior with confidence.

---

<div align="center">

## What's Being Built

</div>

```
calculus/
│
├── derivatives and partial derivatives
├── gradients and directional derivatives
├── Jacobians
├── chain rule by hand
└── backpropagation and numerical checks
```

---

<div align="center">

## The Principles

</div>

**Write the scalar version first.**  
The vector form is easier once the one-dimensional case is clear.

**Expand only when needed.**  
Move from vectors to matrices after the smaller case is solid.

**Match the math to code.**  
The result should line up with a tiny JAX example.

**Trust the gradient path, not the symbolism.**  
If backprop is off by one step, the notation will not save it.

---

<div align="center">

## Current Focus

<!-- Update this section as the work changes -->

```
┌─────────────────────────────────────────────────────────┐
│  CALCULUS — ACTIVE                                      │
│                                                         │
│  Building: chain rule intuition and gradient flow      │
│  Reading:  how loss functions produce usable signals   │
│  Goal:     make training behavior easier to debug      │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## Why It Matters

Calculus is the part of the repo that explains how learning actually happens.

- Loss functions become useful through gradients.
- Optimizers only make sense if the derivative path is clear.
- Training bugs often show up as broken gradient flow.
- Stable implementations depend on knowing where the math is sensitive.

---

## Working Style

- Start with the scalar version.
- Expand to vectors and matrices.
- Check the symbolic result against a tiny JAX example.
- Only then trust it inside a bigger training loop.

---

## Next Step

Keep a set of derivations that I can use as reference while building the training loop and transformer code.