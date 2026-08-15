# 12 — Projects, Papers & Resources

> Theory becomes skill through building. Twenty-four graded projects, the papers worth reading,
> and how to prepare for interviews.

---

## Table of contents

- [How to do projects that count](#how-to-do-projects-that-count)
- [Tier 1 — Foundations (from scratch, NumPy only)](#tier-1--foundations-from-scratch-numpy-only)
- [Tier 2 — Applied classical ML](#tier-2--applied-classical-ml)
- [Tier 3 — Deep learning](#tier-3--deep-learning)
- [Tier 4 — Specialization](#tier-4--specialization)
- [Tier 5 — Production systems](#tier-5--production-systems)
- [Must-read papers](#must-read-papers)
- [Books](#books)
- [Courses](#courses)
- [Interview preparation](#interview-preparation)
- [Staying current](#staying-current)

---

## How to do projects that count

**A portfolio project is not a tutorial you followed.** The difference a hiring manager sees:

| Weak project | Strong project |
|--------------|---------------|
| Titanic / MNIST / Iris | A dataset you collected or a problem you found |
| "98% accuracy!" | "Baseline was 91%; my model gets 94.2% ± 0.8; here's the error analysis" |
| A notebook | A repo with a README, tests, and a deployed endpoint |
| No evaluation discussion | A justified metric, a validation scheme, and known failure modes |
| Works once | Reproducible: pinned deps, seeds, one-command run |

**Every project README should answer:** What problem? What data (and where from)? What
baseline? What metric and why? What did you try and what happened? What are the limitations?
How do I run it?

**The error analysis section is what separates you from everyone else.** Almost nobody writes
it, and it's the clearest signal of real skill.

---

## Tier 1 — Foundations (from scratch, NumPy only)

No `sklearn`, no PyTorch. The point is to prove you understand the mechanics.

**1. Linear regression from scratch**
Implement both the closed-form normal equations and gradient descent. Compare their solutions
and runtimes as $d$ grows. Add L2 regularization and show it fixes a deliberately collinear
design matrix.
*Validates:* [01 §1](01_math_foundations.md#part-1--linear-algebra), [03 §1](03_supervised_learning.md#1-linear-regression)

**2. Logistic regression from scratch**
Sigmoid, binary cross-entropy, analytic gradient. Verify your gradient with finite differences.
Plot the decision boundary. Show what happens with perfectly separable data and no
regularization.
*Validates:* [03 §2](03_supervised_learning.md#2-logistic-regression)

**3. Neural network from scratch**
A 2-layer MLP with ReLU, softmax, and full backpropagation in NumPy. Train on MNIST.
**Match PyTorch's gradients to within $10^{-6}$** — that's the real deliverable.
*Validates:* [05 §4](05_deep_learning.md#4-backpropagation)

**4. K-Means and PCA from scratch**
Implement Lloyd's algorithm with k-means++ init. Implement PCA via SVD. Show empirically that
$J$ decreases monotonically, and that your PCA matches `sklearn` up to sign flips.
*Validates:* [04 §1](04_unsupervised_learning.md#1-k-means-clustering), [04 §5](04_unsupervised_learning.md#5-principal-component-analysis-pca)

**5. Decision tree from scratch**
Recursive splitting with Gini impurity, `max_depth`, and `min_samples_leaf`. Then wrap it in
bagging to build a random forest and demonstrate the variance reduction empirically.
*Validates:* [03 §6–7](03_supervised_learning.md#6-decision-trees)

**6. Optimizer comparison**
Implement SGD, momentum, RMSProp, and Adam. Race them on the Rosenbrock function and on a real
training run. Visualize the trajectories on a 2-D loss surface.
*Validates:* [01 §5](01_math_foundations.md#part-5--optimization)

---

## Tier 2 — Applied classical ML

**7. End-to-end tabular prediction**
Pick a real dataset (not Titanic). Full workflow: EDA → baseline → feature engineering →
gradient boosting → tuning → error analysis. **Deliverable: a written report on what moved the
metric and what didn't.**

**8. Imbalanced classification**
Fraud or rare-disease detection. Compare class weights, threshold tuning, SMOTE, and focal loss.
Show that accuracy is useless here and PR-AUC isn't. Calibrate the probabilities and show a
reliability diagram before and after.
*Validates:* [02 §7, §9](02_ml_fundamentals.md#7-evaluation-metrics)

**9. Time-series forecasting**
Build a naive baseline, then ARIMA, then LightGBM on lag features. Use proper `TimeSeriesSplit`.
**Explicitly demonstrate a leakage bug and then fix it** — show the inflated score and the
honest one.
*Validates:* [03 §12](03_supervised_learning.md#12-time-series)

**10. Customer segmentation**
K-Means, GMM, DBSCAN, and HDBSCAN on the same data. Compare with silhouette and by *interpreting*
the clusters. Produce named personas with distinguishing statistics, not just cluster IDs.
*Validates:* [04](04_unsupervised_learning.md)

**11. Feature selection study**
On a high-dimensional dataset, compare filter, wrapper, embedded (lasso), permutation
importance, and SHAP. Show where they disagree and explain why (hint: correlated features).

**12. A/B test analysis**
Design and analyze an experiment: power analysis for the sample size, the appropriate test,
confidence intervals, and multiple-comparison correction. Write the decision memo.
*Validates:* [01 §4](01_math_foundations.md#part-4--statistics)

---

## Tier 3 — Deep learning

**13. Image classifier with transfer learning**
Fine-tune a pretrained ResNet or ConvNeXt on a custom dataset (photograph it yourself — 10
classes, 100 images each). Compare feature extraction vs full fine-tuning. Ablate the
augmentation policy.

**14. Train a small GPT from scratch**
~10M parameters on TinyShakespeare. Implement tokenization, the transformer block, causal
masking, and generation with temperature/top-p. Watch the loss curve and generate samples at
checkpoints to see coherence emerge.
*Validates:* [05 §9](05_deep_learning.md#9-attention-and-transformers), [06 §4](06_nlp_and_llms.md#4-language-modelling)

**15. Autoencoder and VAE**
Train both on MNIST or Fashion-MNIST. Visualize the latent space in 2-D. Interpolate between
two points and show that the VAE's latent space is continuous while the plain autoencoder's
isn't. Generate new samples.
*Validates:* [04 §8](04_unsupervised_learning.md#8-autoencoders)

**16. Semantic segmentation with U-Net**
Train on a small segmentation dataset. Compare cross-entropy, Dice, and their combination.
Report mIoU and visualize the failures.
*Validates:* [07 §5](07_computer_vision.md#5-segmentation)

**17. Object detection**
Fine-tune YOLO on a custom dataset you label yourself. Understand the box format conversions,
NMS, and mAP. Analyze performance by object size.
*Validates:* [07 §4](07_computer_vision.md#4-object-detection)

**18. Reinforcement learning agent**
Tabular Q-learning on FrozenLake (extending `reinforcement/q_learning_frozenlake_example.py`),
then DQN on CartPole, then PPO on LunarLander. Plot learning curves **over 5 seeds** and show
the variance.
*Validates:* [08](08_reinforcement_learning.md)

---

## Tier 4 — Specialization

**19. RAG system with a real eval suite**
Ingest a document corpus you care about. Hybrid retrieval (BM25 + embeddings) with RRF fusion
and a cross-encoder reranker. **Build a 50-question eval set and measure retrieval recall
separately from answer faithfulness.** Show the improvement from reranking numerically.
*Validates:* [06 §8, §10](06_nlp_and_llms.md#8-retrieval-augmented-generation-rag)

**20. Fine-tune an LLM with LoRA**
Curate ~1,000 high-quality instruction pairs for a specific task. Fine-tune with QLoRA.
Evaluate against the base model with both automatic metrics and human/LLM judging. Report the
parameter count and memory saved.
*Validates:* [06 §6](06_nlp_and_llms.md#6-adapting-a-pretrained-model)

**21. Multimodal search**
Build image search using CLIP embeddings: text→image and image→image, with an ANN index.
Measure recall@k against a labeled subset.
*Validates:* [07 §7](07_computer_vision.md#clip--connecting-vision-and-language)

**22. Reproduce a paper**
Pick a paper with a clear, modest result. Implement it from the paper alone (read the code only
after you're stuck for a day). Document every discrepancy between the paper and what actually
worked. **This is the single most valuable exercise for a research track.**

---

## Tier 5 — Production systems

**23. Deployed model with full MLOps**
A model behind a FastAPI endpoint, containerized, with: input validation, structured logging,
a health check, unit + data + behavioural tests in CI, an experiment tracker, a monitoring
dashboard, drift detection, and a documented rollback procedure. **Include the training/serving
skew test.**
*Validates:* [09](09_mlops_and_deployment.md)

**24. Capstone — an end-to-end system**
Your own problem, start to finish: data collection, modelling, deployment, monitoring, and a
written design doc explaining every significant choice and its alternatives. This is your
portfolio centerpiece.

---

## Must-read papers

### Foundational
| Paper | Year | Why |
|-------|------|-----|
| *Learning representations by back-propagating errors* (Rumelhart, Hinton, Williams) | 1986 | Backpropagation |
| *Support-Vector Networks* (Cortes & Vapnik) | 1995 | SVMs and the kernel trick |
| *Random Forests* (Breiman) | 2001 | Bagging and ensembles |
| *Greedy Function Approximation* (Friedman) | 2001 | Gradient boosting |
| *A Few Useful Things to Know about ML* (Domingos) | 2012 | The best short essay on practical ML |

### Deep learning
| Paper | Year | Why |
|-------|------|-----|
| *ImageNet Classification with Deep CNNs* (AlexNet) | 2012 | Started the deep learning era |
| *Dropout* (Srivastava et al.) | 2014 | Regularization |
| *Batch Normalization* (Ioffe & Szegedy) | 2015 | Trainability |
| *Deep Residual Learning* (ResNet) | 2015 | Residual connections; enabled real depth |
| *Adam* (Kingma & Ba) | 2014 | The default optimizer |
| *Auto-Encoding Variational Bayes* (VAE) | 2013 | The reparameterization trick |
| *Generative Adversarial Networks* (Goodfellow et al.) | 2014 | Adversarial training |

### Transformers and LLMs
| Paper | Year | Why |
|-------|------|-----|
| ***Attention Is All You Need*** | 2017 | **The most important ML paper of the last decade** |
| *BERT* | 2018 | Bidirectional pretraining |
| *Language Models are Few-Shot Learners* (GPT-3) | 2020 | In-context learning, scale |
| *Scaling Laws for Neural LMs* (Kaplan et al.) | 2020 | The scaling relationship |
| *Training Compute-Optimal LLMs* (Chinchilla) | 2022 | Corrected the scaling recipe |
| *LoRA* | 2021 | Parameter-efficient fine-tuning |
| *InstructGPT* (Ouyang et al.) | 2022 | RLHF |
| *Direct Preference Optimization* | 2023 | Alignment without RL |
| *Chain-of-Thought Prompting* | 2022 | Reasoning via intermediate steps |
| *FlashAttention* | 2022 | Made long context practical |
| *Retrieval-Augmented Generation* (Lewis et al.) | 2020 | RAG |

### Vision
| Paper | Year | Why |
|-------|------|-----|
| *U-Net* | 2015 | Segmentation; also the diffusion backbone |
| *Faster R-CNN* | 2015 | Two-stage detection |
| *YOLO* | 2015 | Real-time detection |
| *Focal Loss* (RetinaNet) | 2017 | Extreme class imbalance |
| *An Image is Worth 16x16 Words* (ViT) | 2020 | Transformers for vision |
| *CLIP* | 2021 | Vision–language alignment |
| *Denoising Diffusion Probabilistic Models* | 2020 | Modern generative imaging |
| *Latent Diffusion* (Stable Diffusion) | 2022 | Made it affordable |

### RL
| Paper | Year | Why |
|-------|------|-----|
| *Human-level control through deep RL* (DQN) | 2015 | Deep RL |
| *Proximal Policy Optimization* | 2017 | The workhorse algorithm |
| *Soft Actor-Critic* | 2018 | Continuous control |
| *Mastering the game of Go* (AlphaGo) | 2016 | Search + learning |

### Systems and practice
| Paper | Year | Why |
|-------|------|-----|
| *Hidden Technical Debt in ML Systems* (Sculley et al.) | 2015 | Why ML systems rot |
| *Beyond Accuracy: Behavioral Testing (CheckList)* | 2020 | How to test models |
| *Model Cards for Model Reporting* | 2019 | Documentation standards |
| *On the Dangers of Stochastic Parrots* | 2021 | LLM risk critique |

**How to read a paper (the three-pass method):**
1. **Pass 1 (5 min):** Title, abstract, figures, conclusion. Decide if it's worth more.
2. **Pass 2 (1 hr):** Full read, skipping proofs. Note what's actually novel vs. context.
3. **Pass 3 (4+ hrs):** Reimplement the core idea. You only truly understand a paper you've
   implemented.

---

## Books

| Book | Level | Best for |
|------|-------|----------|
| *Hands-On ML with Scikit-Learn, Keras & TensorFlow* — Géron | Beginner→Inter. | **The best first book.** Practical, code-first |
| *An Introduction to Statistical Learning* (ISL) — James et al. | Beginner→Inter. | Statistical intuition; free PDF |
| *The Elements of Statistical Learning* (ESL) — Hastie et al. | Advanced | The classical ML reference; free PDF |
| *Pattern Recognition and Machine Learning* — Bishop | Advanced | Bayesian perspective; rigorous |
| *Deep Learning* — Goodfellow, Bengio, Courville | Intermediate→Adv. | The DL reference; free online |
| *Dive into Deep Learning* (d2l.ai) | Beginner→Adv. | Interactive, code + math together; free |
| *Reinforcement Learning: An Introduction* — Sutton & Barto | Intermediate | **The** RL book; free PDF |
| *Mathematics for Machine Learning* — Deisenroth et al. | Beginner→Inter. | Exactly the math you need; free PDF |
| *Designing Machine Learning Systems* — Chip Huyen | Intermediate | **The best MLOps book** |
| *Probabilistic Machine Learning* — Murphy | Advanced | Comprehensive modern reference |
| *Speech and Language Processing* — Jurafsky & Martin | Intermediate | NLP; free drafts |
| *Build a Large Language Model (From Scratch)* — Raschka | Intermediate | LLMs by implementation |

---

## Courses

| Course | Provider | Notes |
|--------|----------|-------|
| **Machine Learning Specialization** — Andrew Ng | Coursera | The canonical starting point |
| **Deep Learning Specialization** — Andrew Ng | Coursera | Excellent intuition for DL |
| **Practical Deep Learning for Coders** — fast.ai | Free | Top-down, code-first; superb |
| **CS231n: CNNs for Visual Recognition** | Stanford | The best CV course; notes are free |
| **CS224n: NLP with Deep Learning** | Stanford | The best NLP course |
| **Neural Networks: Zero to Hero** — Karpathy | Free (YouTube) | **Build GPT from scratch. Do this one.** |
| **Spinning Up in Deep RL** — OpenAI | Free | The clearest RL introduction |
| **Full Stack Deep Learning** | Free | Production ML |
| **MLOps Zoomcamp** — DataTalks | Free | Hands-on MLOps |
| **Mathematics for ML** — Imperial | Coursera | Math foundations |

---

## Interview preparation

### The five rounds you'll face

**1. Coding** — LeetCode-style plus ML-specific implementations. Practice: implement K-Means,
kNN, logistic regression with gradient descent, and a simple neural net forward/backward pass
on a whiteboard.

**2. ML theory** — Expect the [self-check questions](00_learning_paths.md#self-assessment-checkpoints)
from Module 00 nearly verbatim. Bias–variance, regularization, why cross-entropy, how backprop
works, what happens with imbalanced classes.

**3. ML system design** — The differentiator for senior roles. Structure your answer:

```
1. CLARIFY    Scale? Latency budget? Online or batch? What's the actual decision?
2. METRICS    Offline metric, online metric, guardrails. What does failure cost?
3. DATA       Sources, labels, volume, freshness, how labels arrive (and how late)
4. FEATURES   What signals exist; how they're computed at train and serve time
5. MODEL      Baseline first, then the real proposal. Justify the complexity.
6. EVALUATION Validation scheme, offline→online gap, A/B design
7. SERVING    Architecture, latency, scaling, caching, fallbacks
8. MONITORING Drift, degradation, retraining triggers
9. TRADEOFFS  What you'd do differently with 10× the data / 1/10 the latency budget
```

Classic prompts: design a recommendation system for a video platform, a fraud detection system,
a search ranking system, a feed ranking system, an LLM-powered support assistant.

**4. Behavioural / project deep-dive** — Be ready to go three levels deep on anything in your
resume. "Why that model?" "What did you try that failed?" "What would you do differently?"
**Have a story about a project that failed and what you learned** — this question is nearly
universal.

**5. Take-home** — Prioritize: a clear README, a sound validation scheme, an honest baseline
comparison, and error analysis. Do *not* chase the last 0.5% of accuracy; reviewers are grading
your process and judgment.

### The most common interview questions

1. Explain the bias–variance tradeoff.
2. How do you handle imbalanced data?
3. L1 vs L2 regularization — what's the difference and why?
4. How does gradient boosting differ from random forest?
5. When would you *not* use deep learning?
6. Explain how a transformer works.
7. Your model performs well offline but poorly in production. Walk me through your debugging.
8. How would you detect data leakage?
9. Which metric would you use for [scenario] and why?
10. How do you know when to stop training?

### Red flags you should avoid giving off
Quoting accuracy on imbalanced data; not knowing your own project's baseline; saying "I'd use a
neural network" for 500 rows of tabular data; never mentioning validation strategy; being unable
to explain a model you listed on your resume.

---

## Staying current

**The field moves fast, but the fundamentals don't.** Modules 01–05 of this guide will still be
accurate in a decade. Spend 80% of your time there and 20% tracking developments.

| Source | Type |
|--------|------|
| **Papers with Code / alphaXiv** | Papers + implementations |
| **arXiv sanity / Hugging Face Daily Papers** | Filtered paper feeds |
| **The Batch** (deeplearning.ai) | Weekly newsletter |
| **Import AI** (Jack Clark) | Weekly, policy-aware |
| **Sebastian Raschka's Ahead of AI** | Deep technical explainers |
| **Lilian Weng's blog** | Best-in-class technical deep dives |
| **Andrej Karpathy** (YouTube, blog) | Fundamentals, taught superbly |
| **Hugging Face blog & docs** | Practical, current |
| **r/MachineLearning** | Community discussion |

**A sustainable weekly habit:** one newsletter (20 min), one paper skimmed (20 min), one thing
implemented (2 hrs). That beats doomscrolling AI Twitter by an enormous margin.

---

## Final word

The gap between people who *know about* machine learning and people who can *do* machine
learning is almost entirely in Modules 01 and 02 — the mathematics and the fundamentals — plus
the habit of building things end to end and looking honestly at what went wrong.

Most people skip the math because it's slow, and skip error analysis because it's uncomfortable.
Doing both is the whole path from zero to hero.

Now go build something.

---

**Back to:** [Guide index](README.md) · [Learning paths](00_learning_paths.md)
