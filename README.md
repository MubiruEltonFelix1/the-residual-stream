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
├── 🧮  foundations/          Mathematical bedrock. Working code, not just theory.
│   ├── linear_algebra/       Vectors, matrices, projections, SVD
│   ├── calculus/             Derivatives, Jacobians, backprop by hand
│   ├── probability/          Distributions, KL divergence, Bayesian reasoning
│   └── ml_prerequisites/     ML prerequisites with runnable notebooks
│
├── ⚙️  implementations/      Papers → running code. JAX only. No shortcuts.
│   ├── 01_bpe_tokenizer/     Byte Pair Encoding from the 1994 paper
│   ├── 02_transformer/       Attention Is All You Need (in progress)
│   ├── 03_training_loop/     Forward pass → loss → backprop → Adam
│   ├── 04_lora/              Low-rank adaptation from Hu et al. 2021 (planned)
│   └── 05_scaling_laws/      Chinchilla scaling laws (planned)
│
├── 🔬  experiments/          Original questions. Stated hypotheses. Real results.
│   ├── 01_tokenization_tax/  Does Luganda really cost 3× more tokens?
│   └── 02_warmup_ablation/   How bad is training without LR warmup? (planned)
│
├── 📄  paper_notes/          Structured notes on every paper read
├── 📓  weekly_builds/        One complete, working thing per week
├── 📖  lectures_exercises/   Lecture notes and JAX challenge work
│   ├── lectures/             Course lecture material
│   └── exercises/            Notebook-based practice and challenges
│
├── 📁  research_papers/      Paper PDFs and primary reading material
└── 📦  scratch/              Throwaway scripts, one-off explorations, orphaned experiments
```

---

## Directory Guide — What Lives Where

The tree above tells you what's in each folder. This section tells you *why* a thing lives in one folder instead of another. The pairs that sound similar are not redundant — they serve different stages of the same pipeline.

---

### foundations vs implementations vs experiments

The three folders that form the core workflow:

| Stage | Question it answers | Example content | When to look here |
|---|---|---|---|
| **foundations/** | *Do I understand the math?* | Backpropagation by hand, SVD, Jacobian derivations | Before you write any code. If a later implementation breaks and you suspect the math, come back here. |
| **implementations/** | *Can I build it from the paper?* | BPE tokenizer, transformer blocks, training loop | When you're coding. Each subfolder is one paper turned into runnable JAX code. |
| **experiments/** | *What happens if I change this?* | Tokenization tax across languages, warmup ablation study | After the implementation works. State a hypothesis, run it, write up what you found — including negative results. |

**The pipeline:** You learn the math in `foundations/`. You build the thing in `implementations/`. You push on it and break it in `experiments/`. A notebook that derives backprop belongs in `foundations/`. A notebook that trains a BPE tokenizer belongs in `implementations/`. A notebook that measures how many tokens Luganda needs compared to English belongs in `experiments/` — it starts with a hypothesis, not a tutorial.

If something could go in two places, ask: *am I learning the tool or am I testing a question?* Learning → `implementations/`. Testing → `experiments/`.

---

### paper_notes vs research_papers

| Folder | What it holds | Why separate |
|---|---|---|
| **research_papers/** | The raw PDFs. The source documents. | This is your reference library. You download a PDF and it lives here, unchanged, so you can always find the original. |
| **paper_notes/** | Your own structured notes on what you read. | This is what you extracted from the PDF. The one-sentence claim, the key experiment, what confused you, what you'd test next. |

The rule: `research_papers/` is the text as written by the authors. `paper_notes/` is the text as understood by you. You should be able to throw away the PDF and rebuild the implementation from your notes. If you can't, the note isn't complete.

---

### scratch vs weekly_builds

| Folder | When something goes here | What it is |
|---|---|---|
| **scratch/** | You wrote a one-off script to test an idea, probe an API, or visualize something. You're not sure it'll be useful again. | Throwaway explorations. Quick plots, debug scripts, tangled notebooks. No commitment to polish or document. |
| **weekly_builds/** | You finished a coherent piece of work — an implementation that runs, an experiment that returned results, a milestone reached. | Polished weekly summaries. One folder per week with a README explaining what was built, what broke, and what you learned. |

Think of `scratch/` as a workbench — things get built, torn apart, and sometimes thrown away. `weekly_builds/` is a gallery — only finished (or honestly failed) work that tells a story. If you're wondering whether a file is `scratch/` or `weekly_builds/` material, ask: *would I want to show this to someone as a record of progress?* Yes → `weekly_builds/`. No → `scratch/`.

---

### lectures_exercises

This folder holds outside course material — lecture slides, notes, and challenge exercises that come from a structured program (in this case, an MIT CSAIL masterclass series). It differs from the other folders:

- **It is not original work.** The `.docx` files and the JAX challenge notebook were given or assigned; this folder stores them for reference.
- **It is not organized by topic.** Unlike `foundations/` or `implementations/`, the content here follows the external course schedule, not the repo's internal learning arc.
- **It exists as a supplement.** If a lecture covers something you later implement, the connection should flow through your own code in `implementations/`, not through the lecture slide.

In short: `lectures_exercises/` is what someone else taught you. Everything else in this repo is what you built from it.

---

## The 8-Week Arc


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

> Full weekly updates logged in [`weekly_builds/`](./weekly_builds/)

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
