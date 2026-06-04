<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=the-residual-stream&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=building%20at%20the%20frontier%2C%20in%20public%2C%20from%20first%20principles&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<!-- BADGES ROW 1 — STATUS -->
![Status](https://img.shields.io/badge/status-actively_building-22c55e?style=for-the-badge&labelColor=0f0c29)
![Sprint](https://img.shields.io/badge/sprint-8_week_frontier_arc-6366f1?style=for-the-badge&labelColor=0f0c29)
![From](https://img.shields.io/badge/from-Kampala,_Uganda-f59e0b?style=for-the-badge&labelColor=0f0c29&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyeiIvPjwvc3ZnPg==)

<!-- BADGES ROW 2 — STACK -->
![JAX](https://img.shields.io/badge/JAX-from_scratch-a78bfa?style=flat-square&logo=python&logoColor=white&labelColor=1e1b4b)
![Papers](https://img.shields.io/badge/papers_read-building_the_list-38bdf8?style=flat-square&labelColor=0c4a6e)
![Commits](https://img.shields.io/badge/commits-daily-34d399?style=flat-square&labelColor=064e3b)
![License](https://img.shields.io/badge/license-MIT-f472b6?style=flat-square&labelColor=4a044e)

<br/>

<!-- TYPING ANIMATION -->
[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=18&pause=1000&color=A78BFA&center=true&vCenter=true&width=700&lines=Implementing+transformers+from+scratch+in+JAX;Reading+the+papers+that+built+the+field;Closing+the+gap+between+knowing+and+building;East+Africa+%E2%86%92+Frontier+Lab+Research+Intern)](https://git.io/typing-svg)

</div>

---

## The Setup

It's 11 PM somewhere in Kampala.

A student opens a terminal. No Jupyter tutorial. No course to follow. Just a blank `.py` file, the Chinchilla paper, and a question that won't leave them alone:

> *"I understand how transformers work. But can I actually build one — from the matrix multiply up — that trains, that converges, that I can debug when it breaks?"*

That's what this repo is. Not a collection of finished projects. A **live record** of the gap between understanding something and being able to build it — closing, week by week, in public.

The target: frontier ML research intern level. The method: implement everything from scratch, read the primary literature cold, run real experiments, and write up what was found — including what broke.

This is the residual stream. Every week, the weights update.

---

<div align="center">

## What's Being Built

</div>

```
the-residual-stream/
│
├── 🧮  foundations/          Mathematical bedrock. No ML yet.
│   ├── linear_algebra/       SVD, eigendecomposition, projections
│   ├── calculus/             Jacobians, backprop by hand, chain rule
│   └── probability/          KL divergence, MLE, Bayesian reasoning
│
├── ⚙️  implementations/      Papers → running code. JAX only. No shortcuts.
│   ├── 01_bpe_tokenizer/     Byte Pair Encoding from the 1994 paper
│   ├── 02_transformer/       Attention Is All You Need, implemented
│   ├── 03_training_loop/     Forward pass → loss → backprop → Adam
│   ├── 04_lora/              Low-rank adaptation from Hu et al. 2021
│   └── 05_chinchilla_laws/   The scaling calculator that broke GPT-3
│
├── 📄  paper_notes/          Structured notes on every paper read
│   ├── template.md           The format used for every paper
│   └── [paper].md            Claim · Key experiment · What I'd test next
│
├── 🔬  experiments/          Original questions. Stated hypotheses. Real results.
│   ├── 01_tokenization_tax/  Does Luganda really cost 3× more tokens?
│   └── 02_warmup_ablation/   How bad is training without LR warmup?
│
├── 📓  weekly_builds/        One complete, working thing per week
│
└── 📋  LEARNING_LOG.md       Honest weekly updates. What broke. What I don't know yet.
```

---

<div align="center">

## The 8-Week Arc

</div>

<table>
<tr>
<td width="50%">

**Weeks 1–2 — Foundations**
- [ ] Backpropagation by hand (scalar → vector → matrix)
- [ ] BPE tokenizer from scratch in JAX
- [ ] Cross-entropy loss, stable log-softmax
- [ ] Adam optimizer: implement the paper exactly
- [ ] nanoGPT re-implemented alone, no looking

</td>
<td width="50%">

**Weeks 3–4 — Read the Canon**
- [ ] *Attention Is All You Need* — Vaswani et al.
- [ ] *GPT-3: Language Models are Few-Shot Learners*
- [ ] *Chinchilla* — Hoffmann et al. 2022
- [ ] *LoRA* — Hu et al. 2021
- [ ] *Constitutional AI* — Anthropic 2022

</td>
</tr>
<tr>
<td width="50%">

**Weeks 5–6 — Original Experiments**
- [ ] Tokenization tax: English vs Luganda (hypothesis first)
- [ ] Warmup ablation study on a training run
- [ ] Reproduce a result from a paper, then extend it
- [ ] Write up findings in experiment READMEs

</td>
<td width="50%">

**Weeks 7–8 — Push the Frontier**
- [ ] LoRA implementation on a small LM
- [ ] Something I haven't decided yet *(following curiosity)*
- [ ] Clean every README, every notebook
- [ ] Write the full retrospective

</td>
</tr>
</table>

---

<div align="center">

## The Principles

</div>

**Build before you explain.**  
Every folder in `implementations/` started as a blank file, not a copy of someone else's code. The point is not the output. The point is the hour where nothing works and you have to figure out why.

**State the hypothesis before running the experiment.**  
Every folder in `experiments/` has a `README.md` written before the notebook. What am I expecting? Why? What would change my mind? This is what research actually is.

**Commit the confusion, not just the clarity.**  
`LEARNING_LOG.md` tracks what I don't understand yet. A portfolio that only shows finished, polished work is a highlight reel. This is the full game.

**Read the source, not the summary.**  
Every paper in `paper_notes/` was read in full — abstract, methods, experiments, limitations. Blog posts are for after.

---

<div align="center">

## Currently In The Stream

<!-- Update this section every single week -->

```
┌─────────────────────────────────────────────────────────┐
│  WEEK 1 — ACTIVE                                        │
│                                                         │
│  Building: BPE tokenizer + full training loop in JAX    │
│  Reading:  Vaswani et al. 2017                          │
│  Blocked on: causal masking shape in multi-head attn    │
│  Resolved: [in progress]                                │
└─────────────────────────────────────────────────────────┘
```

> Full weekly updates in [`LEARNING_LOG.md`](./LEARNING_LOG.md)

</div>

---

<div align="center">

## The Stack

</div>

<div align="center">

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-9333ea?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

**The rule:** implementations use only JAX and NumPy. No PyTorch. No HuggingFace. No library that hides what's actually happening. The constraint is the point.

---

<div align="center">

## The Paper Notes Format

Every paper gets the same structure. No exceptions.

</div>

```markdown
## [Paper Title] — [Year]

**The one-sentence claim** (in my own words, not the abstract's)

**Why it matters** — what changed after this paper?

**The key experiment** — the single result that makes or breaks the argument

**What I found non-obvious** — the thing that confused me and how I resolved it

**What I'd test next** — if I had compute, what experiment would I run?

**Implementation** — link to code if I built something from this paper
```

The last two sections are the ones that matter. Reading comprehension is table stakes. Research taste is the thing being built.

---

<div align="center">

## Benchmarks I'm Measuring Myself Against

</div>

These are the questions a frontier lab research intern should be able to answer cold. Checking these off as the answer becomes automatic — not recited.

| Question | Status |
|----------|--------|
| Implement backprop by hand for a 2-layer network | 🔄 in progress |
| Explain why Adam converges faster than SGD with momentum | 🔄 in progress |
| Read a JAX training loop and spot a bug in 5 minutes | ⬜ not yet |
| Explain what a gradient checkpoint is and when to use it | ⬜ not yet |
| Derive attention from query-key-value formulation | ⬜ not yet |
| Explain the Chinchilla optimal ratio without notes | ⬜ not yet |
| Implement LoRA and explain the rank decomposition | ⬜ not yet |
| Read a new paper and identify the weakest assumption | ⬜ not yet |

---

<div align="center">

## Why In Public

</div>

There's a version of this where the work happens privately, and a polished portfolio appears when it's "ready."

That version is comfortable and useless.

Building in public creates a forcing function. The README has to be clear because someone else might read it. The experiment has to have a real hypothesis because it's written down before running. The learning log has to be honest because there's no point otherwise.

It also means that if someone working on similar problems finds this repo — at Anthropic, DeepMind, Masakhane, or a research group at Makerere — there's something real to look at, engage with, and push back on.

That's the goal. Not the performance of learning. The actual thing.

---

<div align="center">

## Get In Touch

If something here is **wrong**, I want to know.  
If you're working on **similar problems**, I want to talk.  
If you're at a lab and something here is **interesting**, reach out.

<br/>

[![Email](https://img.shields.io/badge/email-reach_out-6366f1?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your@email.com)
[![Twitter](https://img.shields.io/badge/twitter-follow_the_build-1d9bf0?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/yourhandle)
[![LinkedIn](https://img.shields.io/badge/linkedin-connect-0077b5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)

<br/>

---

<sub>
Started: June 2026 &nbsp;·&nbsp; Location: Kampala, Uganda &nbsp;·&nbsp;
Inspired by: Prof. Whitmore's MIT CSAIL Masterclass &nbsp;·&nbsp;
Every weight update logged publicly
</sub>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

</div>
