# ai_ml_from_scratch

Learning AI/ML from first principles — a complete curriculum plus runnable example code.

## 📚 [The AI/ML Mastery Guide — Zero to Hero](guide/README.md)

A self-contained curriculum covering every concept from linear algebra to LLM deployment, with
full mathematical derivations, terminology definitions, and structured learning paths.

| Module | Topic |
|--------|-------|
| [00](guide/00_learning_paths.md) | **Learning Paths** — roadmaps, schedules, checkpoints |
| [01](guide/01_math_foundations.md) | **Mathematical Foundations** — linear algebra, calculus, probability, statistics, optimization, information theory |
| [02](guide/02_ml_fundamentals.md) | **ML Fundamentals** — learning theory, bias–variance, losses, validation, metrics |
| [03](guide/03_supervised_learning.md) | **Supervised Learning** — regression, SVM, trees, ensembles, boosting |
| [04](guide/04_unsupervised_learning.md) | **Unsupervised Learning** — clustering, PCA, manifolds, anomaly detection |
| [05](guide/05_deep_learning.md) | **Deep Learning** — backprop derived, CNNs, RNNs, transformers |
| [06](guide/06_nlp_and_llms.md) | **NLP & LLMs** — tokenization, pretraining, LoRA, RLHF, RAG, agents |
| [07](guide/07_computer_vision.md) | **Computer Vision** — detection, segmentation, ViT, diffusion |
| [08](guide/08_reinforcement_learning.md) | **Reinforcement Learning** — MDPs, Q-learning, policy gradients, PPO |
| [09](guide/09_mlops_and_deployment.md) | **MLOps & Deployment** — serving, monitoring, drift, testing |
| [10](guide/10_glossary.md) | **Glossary** — every term, A–Z |
| [11](guide/11_notation.md) | **Notation Reference** — every symbol |
| [12](guide/12_projects_and_resources.md) | **Projects & Resources** — 24 projects, papers, books, interview prep |
| [13](guide/13_frameworks.md) | **Frameworks** — scikit-learn, TensorFlow/Keras, PyTorch: APIs, use cases, sample code |

**New here?** Start with the [learning paths](guide/00_learning_paths.md) to pick a route.

## 💻 Example code

| Directory | Contents |
|-----------|----------|
| `notebooks/` | [**Framework comparison**](notebooks/13_framework_comparison.ipynb) — the same model built in scikit-learn, Keras, and PyTorch side by side, with training curves and exercises |
| `python/` | Library tutorials: NumPy, Pandas, Matplotlib, scikit-learn, TensorFlow, PyTorch, OpenCV, NLTK/spaCy, Transformers, XGBoost/LightGBM, serialization, FastAPI serving |
| `supervised/` | Linear & logistic regression, kNN, SVM, decision tree, random forest, neural network |
| `unsupervised/` | K-Means, PCA, autoencoder |
| `reinforcement/` | Q-learning on FrozenLake |

Each script is standalone and documented, with theory links back to the guide.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python supervised/linear_regression_example.py
```
