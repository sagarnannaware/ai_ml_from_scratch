# 00 — Learning Paths

> How to actually get from zero to hero, in what order, with checkpoints that tell you
> whether you're ready to move on.

---

## Table of contents

- [The four stages](#the-four-stages)
- [Path A — The Complete Path (12 months)](#path-a--the-complete-path-12-months)
- [Path B — Fast Track to Employable (16 weeks)](#path-b--fast-track-to-employable-16-weeks)
- [Path C — Software Engineer → ML Engineer (10 weeks)](#path-c--software-engineer--ml-engineer-10-weeks)
- [Path D — LLM / GenAI Engineer (12 weeks)](#path-d--llm--genai-engineer-12-weeks)
- [Path E — Research Track (18 months)](#path-e--research-track-18-months)
- [Specialization tracks](#specialization-tracks)
- [Study method that works](#study-method-that-works)
- [Self-assessment checkpoints](#self-assessment-checkpoints)
- [Common failure modes](#common-failure-modes)

---

## The four stages

Everyone passes through the same four stages. Naming them helps you know where you are.

| Stage | You can... | You cannot yet... | Typical duration |
|-------|-----------|-------------------|------------------|
| **1. User** | Call `model.fit()`, follow tutorials, read a confusion matrix | Explain *why* the model failed | 1–2 months |
| **2. Practitioner** | Frame a problem, build a validation scheme, tune, deploy | Derive the algorithm or debug it from first principles | 4–8 months |
| **3. Engineer** | Implement algorithms from the paper, profile and scale training, own a system in production | Invent new methods | 8–18 months |
| **4. Researcher / Architect** | Read a paper and reproduce it, identify what's actually novel, design new architectures or training regimes | — | 18 months+ |

**The gap most people never cross is 1 → 2**, and the reason is always the same: they
skipped the mathematics. Stage 2 requires knowing *why* gradient descent moves the way it
does, why regularization works, and what a validation set is actually estimating.

---

## Path A — The Complete Path (12 months)

For someone starting from near-zero with ~10–12 hours/week. This is the recommended path.

### Phase 1 — Foundations (Months 1–3)

| Week | Topic | Module | Deliverable |
|------|-------|--------|-------------|
| 1–2 | Python for data: NumPy arrays, broadcasting, vectorization | `python/01_numpy_basics.py` | Implement matrix multiply, no loops |
| 3–4 | Pandas, data cleaning, EDA, plotting | `python/02–03` | EDA notebook on Titanic + a real messy CSV |
| 5–7 | **Linear algebra**: vectors, matrices, rank, eigen, SVD | [01 §1](01_math_foundations.md#part-1--linear-algebra) | Implement PCA using only `numpy.linalg` |
| 8–9 | **Calculus**: derivatives, gradients, chain rule, Jacobians | [01 §2](01_math_foundations.md#part-2--calculus-and-optimization-foundations) | Hand-derive the gradient of MSE and cross-entropy |
| 10–12 | **Probability & statistics**: distributions, Bayes, MLE, CLT, hypothesis testing | [01 §3–4](01_math_foundations.md#part-3--probability) | Derive linear regression as MLE under Gaussian noise |

**Checkpoint 1:** On paper, derive $\hat\beta = (X^\top X)^{-1}X^\top y$ and explain what
goes wrong when $X^\top X$ is singular. If you can't, do not proceed.

### Phase 2 — Classical ML (Months 4–6)

| Week | Topic | Module |
|------|-------|--------|
| 13–14 | Learning theory: risk, generalization, bias–variance, overfitting | [02](02_ml_fundamentals.md) |
| 15–16 | Validation, cross-validation, leakage, metrics (ROC/AUC, PR, F1) | [02 §5–7](02_ml_fundamentals.md#5-model-validation) |
| 17–18 | Linear & logistic regression, regularization (L1/L2/elastic net) | [03 §1–2](03_supervised_learning.md) |
| 19 | kNN, Naive Bayes, LDA/QDA | [03 §3–4](03_supervised_learning.md) |
| 20 | SVM, kernels, the kernel trick | [03 §5](03_supervised_learning.md#5-support-vector-machines-svm) |
| 21–22 | Trees, bagging, random forests | [03 §6–7](03_supervised_learning.md#6-decision-trees) |
| 23–24 | Boosting: AdaBoost, GBM, XGBoost, LightGBM | [03 §8](03_supervised_learning.md#8-boosting) |
| 25–26 | Unsupervised: K-Means, GMM/EM, hierarchical, DBSCAN, PCA, t-SNE/UMAP | [04](04_unsupervised_learning.md) |

**Checkpoint 2:** Implement logistic regression **and** a decision tree from scratch in
NumPy. Beat a `sklearn` baseline on a Kaggle tabular dataset using gradient boosting with
a validation scheme you designed yourself.

### Phase 3 — Deep Learning (Months 7–9)

| Week | Topic | Module |
|------|-------|--------|
| 27–28 | Perceptron → MLP, activation functions, universal approximation | [05 §1–3](05_deep_learning.md) |
| 29–30 | **Backpropagation derived by hand**, autodiff, computational graphs | [05 §4](05_deep_learning.md#4-backpropagation) |
| 31–32 | Optimizers: SGD, momentum, RMSProp, Adam, AdamW, schedules | [05 §5](05_deep_learning.md#5-optimization-in-deep-learning) |
| 33 | Initialization, normalization (batch/layer), dropout, residuals | [05 §6](05_deep_learning.md#6-making-deep-networks-trainable) |
| 34–35 | CNNs: convolution arithmetic, pooling, classic architectures | [05 §7](05_deep_learning.md#7-convolutional-neural-networks-cnns) + [07](07_computer_vision.md) |
| 36–37 | RNN, LSTM, GRU, seq2seq, vanishing gradients | [05 §8](05_deep_learning.md#8-recurrent-neural-networks) |
| 38–39 | **Attention & Transformers** — full derivation | [05 §9](05_deep_learning.md#9-attention-and-transformers) |

**Checkpoint 3:** Implement a 2-layer MLP with backprop in pure NumPy, matching PyTorch
gradients to $10^{-6}$. Then implement single-head self-attention from scratch.

### Phase 4 — Specialize & Ship (Months 10–12)

Pick **one** primary specialization ([tracks below](#specialization-tracks)), plus MLOps.

| Week | Topic |
|------|-------|
| 40–44 | Primary track deep dive (LLMs / CV / RL / tabular-at-scale) |
| 45–47 | MLOps: pipelines, serving, monitoring, drift, CI/CD ([09](09_mlops_and_deployment.md)) |
| 48–52 | **Capstone**: an end-to-end system, deployed, monitored, documented |

**Checkpoint 4 (Hero):** A deployed system with an API, tests, a monitoring dashboard, a
written design doc explaining your modelling choices, and an honest error analysis.

---

## Path B — Fast Track to Employable (16 weeks)

~20 hrs/week. Optimized for getting a first ML job, not for depth. You will have holes;
close them later.

| Weeks | Focus | Non-negotiables |
|-------|-------|-----------------|
| 1–2 | Python + NumPy + Pandas + SQL | Vectorization, joins, window functions |
| 3–4 | Math *minimum*: gradients, matrix multiply, probability, expectation, Bayes | [01](01_math_foundations.md) §1.1–1.6, §2.1–2.4, §3.1–3.6 |
| 5–6 | ML fundamentals + validation + metrics | Never leak data; know when to use PR-AUC over ROC-AUC |
| 7–9 | Supervised: linear/logistic, trees, random forest, **gradient boosting** | XGBoost/LightGBM tuning |
| 10–11 | Unsupervised + feature engineering | PCA, K-Means, target/one-hot encoding |
| 12–13 | Deep learning essentials: MLP, CNN, fine-tuning a pretrained model | PyTorch training loop from memory |
| 14 | LLM basics: embeddings, prompting, RAG | Build a small RAG app |
| 15 | Deployment: FastAPI + Docker + a cloud endpoint | `python/12_fastapi_flask_serving.py` |
| 16 | Portfolio polish + interview prep | 3 projects, README + results, [12](12_projects_and_resources.md) |

---

## Path C — Software Engineer → ML Engineer (10 weeks)

You can already code, test, deploy, and reason about systems. Skip the software parts.

| Weeks | Focus |
|-------|-------|
| 1–2 | Math refresh: linear algebra + gradients + probability ([01](01_math_foundations.md)) |
| 3 | ML fundamentals & the *statistical* mindset: distributions, not asserts ([02](02_ml_fundamentals.md)) |
| 4 | Validation, metrics, leakage, experiment tracking |
| 5 | Gradient boosting mastery (the workhorse of production tabular ML) |
| 6–7 | PyTorch: training loops, datasets, mixed precision, distributed basics ([05](05_deep_learning.md)) |
| 8 | Fine-tuning pretrained models (vision or text) |
| 9–10 | ML systems design: feature stores, batch vs online, drift, rollbacks ([09](09_mlops_and_deployment.md)) |

**Your edge:** most ML people can't build reliable systems. Lean into serving, latency,
reproducibility, and testing — then add modelling depth.

---

## Path D — LLM / GenAI Engineer (12 weeks)

Assumes Python fluency. Assumes you will still learn the math — GenAI without math caps
you at prompt-tinkering.

| Weeks | Focus | Module |
|-------|-------|--------|
| 1–2 | Math minimum: vectors, dot products, softmax, gradients, entropy/KL | [01](01_math_foundations.md) |
| 3 | Neural nets + backprop + PyTorch | [05 §1–6](05_deep_learning.md) |
| 4–5 | **Transformer from scratch** — attention, positional encodings, KV cache | [05 §9](05_deep_learning.md#9-attention-and-transformers) |
| 6 | Tokenization (BPE), embeddings, sampling (temperature, top-k, top-p) | [06 §1–3](06_nlp_and_llms.md) |
| 7 | Pretraining objectives, scaling laws, emergent behaviour | [06 §5](06_nlp_and_llms.md#5-pretraining) |
| 8 | Fine-tuning: SFT, LoRA/QLoRA, PEFT | [06 §6](06_nlp_and_llms.md#6-adapting-a-pretrained-model) |
| 9 | Alignment: RLHF, DPO, reward models | [06 §7](06_nlp_and_llms.md#7-alignment) |
| 10 | RAG: chunking, embeddings, vector DBs, hybrid search, reranking | [06 §8](06_nlp_and_llms.md#8-retrieval-augmented-generation-rag) |
| 11 | Agents, tool use, structured output, evals | [06 §9–10](06_nlp_and_llms.md#9-tool-use-and-agents) |
| 12 | Inference optimization: quantization, batching, vLLM, cost/latency | [06 §11](06_nlp_and_llms.md#11-serving-and-inference-optimization) |

**Hero test:** train a ~10M-parameter GPT on TinyShakespeare from your own code, then
build a RAG system with an eval suite that catches regressions.

---

## Path E — Research Track (18 months)

Path A (months 1–9) then:

| Months | Focus |
|--------|-------|
| 10–12 | Measure-theoretic probability, convex optimization, statistical learning theory (VC dimension, Rademacher complexity, PAC-Bayes) |
| 13–14 | Paper reproduction: pick 3 papers, reimplement from scratch, match reported numbers |
| 15–16 | Deep dive into one subfield; read every paper it cites and every paper citing it |
| 17–18 | Original contribution: an ablation nobody ran, a failure mode nobody documented; write it up |

**Reading discipline:** 3 papers/week — one skim (title→abstract→figures→conclusion), one
careful read, one *implemented*. Keep a notes repo with your derivations.

---

## Specialization tracks

| Track | Core skills | Start with | Typical role |
|-------|-------------|-----------|--------------|
| **LLM / GenAI** | Transformers, fine-tuning, RAG, evals, inference cost | [06](06_nlp_and_llms.md) | GenAI Engineer, Applied AI |
| **Computer Vision** | CNNs, ViT, detection, segmentation, diffusion | [07](07_computer_vision.md) | CV Engineer, Perception |
| **Tabular / Decision Science** | Boosting, feature engineering, causal inference, uplift | [03](03_supervised_learning.md) | DS, Risk, Fraud, Pricing |
| **Reinforcement Learning** | MDPs, policy gradients, PPO, sim-to-real | [08](08_reinforcement_learning.md) | RL Researcher, Robotics |
| **MLOps / Platform** | Pipelines, serving, monitoring, k8s, cost | [09](09_mlops_and_deployment.md) | MLE, ML Platform |
| **Time Series / Forecasting** | ARIMA, state space, hierarchical, deep forecasting | [03 §12](03_supervised_learning.md) | Demand/Capacity forecasting |

---

## Study method that works

1. **Derive before you import.** Any algorithm you can't sketch on paper, you don't know.
2. **Implement once from scratch, then use the library forever.** One NumPy implementation
   teaches more than fifty `fit()` calls.
3. **Spaced repetition on definitions and formulas.** Anki, 15 min/day. The glossary
   ([10](10_glossary.md)) is written to be turned into cards.
4. **The Feynman test.** Explain backprop to a rubber duck without notation. Where you
   stumble is where you don't understand.
5. **Break things deliberately.** Remove normalization, use lr=10, shuffle your labels.
   Learn what each failure *looks* like — that's debugging skill.
6. **Keep a lab notebook.** Every experiment: hypothesis, change, metric, conclusion.
   This is the single habit that separates practitioners from tinkerers.
7. **Read the source.** `sklearn`'s `_logistic.py`, PyTorch's `nn.MultiheadAttention`.
8. **Ratio: 30% reading, 50% coding, 20% writing/explaining.**

### Weekly rhythm (10 hrs)

| Day | Hours | Activity |
|-----|-------|----------|
| Mon | 1.5 | New theory + hand-written derivations |
| Tue | 1.5 | Implement it from scratch |
| Wed | 1.5 | Apply it with a library on real data |
| Thu | 1.5 | Read the relevant paper / library source |
| Sat | 3 | Project work |
| Sun | 1 | Review, Anki, write up what you learned |

---

## Self-assessment checkpoints

Answer without looking anything up. Anything you miss points at the module to revisit.

### Level 1 — Foundations
1. What does $X^\top X$ being singular mean geometrically, and what do you do about it?
2. Why is the gradient the direction of steepest ascent?
3. State Bayes' theorem and identify prior, likelihood, posterior, evidence.
4. What does a p-value actually mean? What does it *not* mean?
5. What is the difference between covariance and correlation?

### Level 2 — Classical ML
6. Decompose expected test error into bias, variance, and noise. Which does bagging reduce?
7. Why does L1 produce sparse solutions and L2 not? (Geometric answer.)
8. Your model has 99% accuracy on a 1%-positive dataset. What do you do?
9. When is ROC-AUC misleading, and what do you use instead?
10. Give three concrete ways data leakage sneaks into a time-series pipeline.

### Level 3 — Deep Learning
11. Derive $\partial L/\partial W^{(1)}$ for a 2-layer MLP with ReLU and cross-entropy.
12. Why does batch norm help? Give the formula, including train/eval difference.
13. Why $\sqrt{d_k}$ in scaled dot-product attention? Show what breaks without it.
14. What exactly does Adam store per parameter, and how much memory is that for 7B params?
15. Explain vanishing gradients in an RNN in terms of a matrix's spectral radius.

### Level 4 — Systems & Frontier
16. Design a real-time fraud detection system: latency budget, features, retraining.
17. Model quality is fine offline but degrading in production. Diagnose systematically.
18. Explain LoRA's parameter count and why it works.
19. Derive the policy gradient theorem.
20. Your LLM app hallucinates 5% of the time. Give five mitigations, ranked by cost.

---

## Common failure modes

| Failure | Symptom | Fix |
|---------|---------|-----|
| **Tutorial hell** | 40 notebooks, no original work | Build something nobody wrote a tutorial for |
| **Math avoidance** | Stuck at "user" stage for a year | Two weeks of [01](01_math_foundations.md), no code |
| **Framework hopping** | TF → PyTorch → JAX → back | Pick PyTorch. Stay 12 months. |
| **Kaggle-only** | Can model, can't ship | Deploy one model with an API and monitoring |
| **Paper collecting** | 200 unread PDFs | 3/week, one implemented |
| **Skipping evaluation** | "It looks good" | Metrics before models — always |
| **Premature deep learning** | 500 rows, 12-layer net | Baseline first: mean, logistic, gradient boosting |

---

**Next:** [01 — Mathematical Foundations →](01_math_foundations.md)
