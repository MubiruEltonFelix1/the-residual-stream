<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=probability&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=distributions%2C%20uncertainty%2C%20and%20the%20math%20behind%20objectives&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

</div>

---

## The Setup

This is where I keep the probability theory I need to understand objectives, uncertainty, and model behavior.

The goal is not to become abstract about statistics. The goal is to make distributions, likelihood, expectation, and divergence feel like tools I can use in actual machine learning code.

---

<div align="center">

## What's Being Built

</div>

```
probability/
│
├── random variables and distributions
├── expectation and variance
├── conditional probability
├── likelihood and maximum likelihood estimation
└── entropy, cross-entropy, and KL divergence
```

---

<div align="center">

## The Principles

</div>

**Define the distribution clearly.**  
If the random variables are vague, the rest of the derivation will be vague too.

**Write the objective explicitly.**  
Expectations and losses should be visible, not implied.

**Check edge cases with small numbers.**  
Probability mistakes often hide in boundary cases.

**Connect the result back to a metric or loss.**  
The theory should point somewhere useful in code.

---

<div align="center">

## Current Focus

<!-- Update this section as the work changes -->

```
┌─────────────────────────────────────────────────────────┐
│  PROBABILITY — ACTIVE                                   │
│                                                         │
│  Building: intuition for likelihood and divergence     │
│  Reading:  how objectives behave in language models    │
│  Goal:     make training targets feel natural          │
└─────────────────────────────────────────────────────────┘
```

</div>

---

## Why It Matters

Probability is the language behind a lot of ML decisions.

- Language modeling is about predicting distributions.
- Loss functions often come from likelihood.
- Evaluation metrics depend on expectation and aggregation.
- Concepts like entropy and KL show up everywhere in training and analysis.

---

## Working Style

- Define the distribution clearly.
- Write out the expectation or objective explicitly.
- Check the edge cases with small numbers.
- Connect the result back to a loss function or metric.

---

## Next Step

Build a compact set of notes that makes the training objectives in the rest of the repo easier to understand.