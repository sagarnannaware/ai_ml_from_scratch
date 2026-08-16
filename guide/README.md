# AI ML from Scratch — Zero to Hero

A complete, self-contained curriculum that takes you from "I know a bit of Python" to
"I can derive, implement, train, evaluate, and deploy modern ML and AI systems."

Every concept is defined in words, then in mathematics, then in code pointers.
Nothing is assumed except high-school algebra and basic Python.

> **Math rendering:** formulas use LaTeX (`$...$` inline, `$$...$$` block).
> These render on **github.com** (desktop and mobile browser) and in VS Code with the
> *Markdown+Math* or *Markdown Preview Enhanced* extension.
> The **GitHub mobile app** does not render LaTeX — it shows the raw source, so formulas
> look like `\mathbf{x}` there. Open the repo in a mobile **browser** instead.

---

## The Guide

| # | Module | What you learn | Est. time |
|---|--------|----------------|-----------|
| 00 | **[Learning Paths](00_learning_paths.md)** | Roadmaps, tracks, weekly schedules, checkpoints | read first |
| 01 | **[Mathematical Foundations](01_math_foundations.md)** | Linear algebra, calculus, probability, statistics, optimization, information theory | 4–8 weeks |
| 02 | **[ML Fundamentals](02_ml_fundamentals.md)** | Learning theory, bias–variance, loss functions, regularization, validation, metrics | 3–4 weeks |
| 03 | **[Supervised Learning](03_supervised_learning.md)** | Linear/logistic regression, SVM, trees, ensembles, boosting, kNN, Naive Bayes | 4–6 weeks |
| 04 | **[Unsupervised Learning](04_unsupervised_learning.md)** | Clustering, PCA/SVD, manifold learning, density estimation, anomaly detection | 3 weeks |
| 05 | **[Deep Learning](05_deep_learning.md)** | Perceptrons, backprop derivation, CNNs, RNNs, attention, Transformers, training craft | 6–10 weeks |
| 06 | **[NLP & Large Language Models](06_nlp_and_llms.md)** | Tokenization, embeddings, GPT/BERT, pretraining, fine-tuning, LoRA, RLHF, RAG, agents | 5–8 weeks |
| 07 | **[Computer Vision](07_computer_vision.md)** | Convolution math, detection, segmentation, ViT, diffusion models | 4–6 weeks |
| 08 | **[Reinforcement Learning](08_reinforcement_learning.md)** | MDPs, Bellman equations, Q-learning, DQN, policy gradients, PPO | 4–6 weeks |
| 09 | **[MLOps & Deployment](09_mlops_and_deployment.md)** | Pipelines, serving, monitoring, drift, scaling, cost, reliability | 3–4 weeks |
| 10 | **[Glossary (A–Z)](10_glossary.md)** | Every term used in this guide, defined precisely | reference |
| 11 | **[Notation Reference](11_notation.md)** | Every symbol used, with meaning and shape | reference |
| 12 | **[Projects, Papers & Resources](12_projects_and_resources.md)** | 24 graded projects, must-read papers, books, interview prep | ongoing |

---

## How this guide is structured

Each module follows the same rhythm so you always know what you're looking at:

1. **Intuition** — the idea in plain English, with an analogy.
2. **Formal definition** — the mathematics, with every symbol named.
3. **Derivation** — where the formula comes from (not just what it is).
4. **Algorithm** — pseudocode you could implement from scratch.
5. **In practice** — hyperparameters, failure modes, what actually matters.
6. **Code** — a pointer to a runnable script in this repo.

---

## Companion code in this repository

The theory here maps directly onto runnable scripts already in the repo:

```
python/                  # Library-by-library tutorials (NumPy → FastAPI serving)
supervised/              # Linear & logistic regression, SVM, trees, forests, kNN, NN
unsupervised/            # K-Means, PCA, autoencoder
reinforcement/           # Q-learning on FrozenLake
```

| Guide section | Runnable script |
|---|---|
| [Linear regression](03_supervised_learning.md#1-linear-regression) | `supervised/linear_regression_example.py` |
| [Logistic regression](03_supervised_learning.md#2-logistic-regression) | `supervised/logistic_regression_example.py` |
| [k-Nearest Neighbours](03_supervised_learning.md#3-k-nearest-neighbours-knn) | `supervised/knn_classifier_example.py` |
| [SVM](03_supervised_learning.md#5-support-vector-machines-svm) | `supervised/svm_classifier_example.py` |
| [Decision trees](03_supervised_learning.md#6-decision-trees) | `supervised/decision_tree_classifier_example.py` |
| [Random forest](03_supervised_learning.md#7-bagging-and-random-forests) | `supervised/random_forest_classifier_example.py` |
| [Neural networks](05_deep_learning.md) | `supervised/neural_network_classifier_example.py` |
| [K-Means](04_unsupervised_learning.md#1-k-means-clustering) | `unsupervised/kmeans_clustering_example.py` |
| [PCA](04_unsupervised_learning.md#5-principal-component-analysis-pca) | `unsupervised/pca_example.py` |
| [Autoencoders](04_unsupervised_learning.md#8-autoencoders) | `unsupervised/autoencoder_example.py` |
| [Q-learning](08_reinforcement_learning.md#6-q-learning) | `reinforcement/q_learning_frozenlake_example.py` |
| [Gradient boosting](03_supervised_learning.md#8-boosting) | `python/10_xgboost_lightgbm.py` |
| [Transformers / LLMs](06_nlp_and_llms.md) | `python/09_huggingface_transformers_llm.py` |
| [Serving](09_mlops_and_deployment.md) | `python/12_fastapi_flask_serving.py` |

### Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## The one-paragraph version of the whole field

Machine learning is **function approximation from data under uncertainty**. You assume
your data $(x, y)$ is drawn from an unknown distribution $P(x, y)$. You pick a family of
functions $f_\theta$ (the *model*), a way to score mistakes $\ell(f_\theta(x), y)$ (the
*loss*), and a way to search $\theta$ (the *optimizer*, almost always gradient descent).
You minimize the average loss on data you have (*empirical risk*) while keeping the model
simple enough that it also works on data you don't have (*generalization*). Everything
else — deep networks, transformers, diffusion, RL — is a choice of $f_\theta$, $\ell$, and
how you get the data.

$$\theta^\* = \arg\min_\theta \underbrace{\frac{1}{n}\sum_{i=1}^n \ell\big(f_\theta(x_i), y_i\big)}_{\text{empirical risk}} + \underbrace{\lambda \, \Omega(\theta)}_{\text{regularization}}$$

Understand that equation completely and the rest of this guide is detail.

---

## License / usage

Written as a study resource. Use it, fork it, extend it.
