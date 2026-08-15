# 02 — Machine Learning Fundamentals

> The concepts that apply to *every* model — from linear regression to GPT. If you only
> deeply learn one module, make it this one. Most production ML failures are failures of
> this module, not of modelling.

---

## Table of contents

1. [What machine learning actually is](#1-what-machine-learning-actually-is)
2. [Types of learning](#2-types-of-learning)
3. [The bias–variance tradeoff](#3-the-biasvariance-tradeoff)
4. [Loss functions](#4-loss-functions)
5. [Model validation](#5-model-validation)
6. [Regularization](#6-regularization)
7. [Evaluation metrics](#7-evaluation-metrics)
8. [Feature engineering & preprocessing](#8-feature-engineering--preprocessing)
9. [Data problems](#9-data-problems)
10. [Hyperparameter tuning](#10-hyperparameter-tuning)
11. [The ML workflow](#11-the-ml-workflow)

---

## 1. What machine learning actually is

**Definition (Mitchell, 1997):** A computer program learns from experience $E$ with respect to
task $T$ and performance measure $P$, if its performance at $T$, as measured by $P$, improves
with $E$.

### The formal setup

There is an unknown joint distribution $P(x, y)$ over inputs and outputs. You observe a
**training set** of $n$ samples drawn i.i.d. from it:

$$\mathcal{D} = \{(\mathbf{x}_1,y_1),\dots,(\mathbf{x}_n,y_n)\} \sim P(x,y)^n$$

You want a function $f:\mathcal{X}\to\mathcal{Y}$ minimizing the **true risk** (expected loss
over the whole distribution):

$$R(f) = \mathbb{E}_{(x,y)\sim P}\big[\ell(f(x), y)\big]$$

You **cannot compute this** — you don't have $P$. So you minimize the **empirical risk**:

$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n}\ell(f(\mathbf{x}_i), y_i)$$

This is **Empirical Risk Minimization (ERM)**, and it is what every supervised learning
algorithm does.

### The central problem: generalization

$\hat R(f)$ is only an estimate of $R(f)$. The **generalization gap** is:

$$\text{gap} = R(f) - \hat{R}(f)$$

Everything in this module exists to keep that gap small. The gap grows with model capacity and
shrinks with data:

$$R(f) \le \hat{R}(f) + O\!\left(\sqrt{\frac{\text{capacity}}{n}}\right)$$

(A schematic of results from statistical learning theory — VC dimension, Rademacher complexity.
The bounds are usually far too loose to use numerically, but the *shape* is the lesson: more
capacity needs more data.)

### Key terminology

| Term | Definition |
|------|-----------|
| **Feature / predictor / covariate** | An input variable, $x_j$ |
| **Label / target / response** | The output to predict, $y$ |
| **Instance / example / sample** | One $(\mathbf{x}, y)$ pair |
| **Design matrix** | $X\in\mathbb{R}^{n\times d}$ — all examples stacked as rows |
| **Hypothesis space** $\mathcal{H}$ | The set of functions the model can represent |
| **Parameters** $\theta$ | Learned from data (weights, biases, split thresholds) |
| **Hyperparameters** | Set *before* training (learning rate, tree depth, $\lambda$) |
| **Inductive bias** | Assumptions built into the model that let it generalize beyond data |
| **Capacity** | How complex a function the model can fit |
| **i.i.d.** | Independent & identically distributed — the assumption underpinning everything |
| **Inference / prediction** | Applying the trained model to new inputs |

> **The most-violated assumption in ML is i.i.d.** Time series, user data with repeat users,
> images from the same patient, geographically clustered data — none are i.i.d. Almost every
> silent validation failure traces back to this.

### No Free Lunch theorem

Averaged over *all* possible problems, every algorithm performs identically. There is no
universally best model. Performance comes from matching the model's inductive bias to the
structure of your specific problem — CNNs assume spatial locality, RNNs assume sequence,
trees assume axis-aligned interactions, linear models assume additivity.

---

## 2. Types of learning

### By supervision

| Type | Data | Goal | Examples |
|------|------|------|----------|
| **Supervised** | $(\mathbf{x}, y)$ labeled | Predict $y$ from $\mathbf{x}$ | Regression, classification |
| **Unsupervised** | $\mathbf{x}$ only | Find structure | Clustering, PCA, density estimation |
| **Self-supervised** | $\mathbf{x}$, labels derived from $\mathbf{x}$ itself | Learn representations | Next-token prediction, masked LM, SimCLR |
| **Semi-supervised** | Few labeled + many unlabeled | Use unlabeled data to help | Pseudo-labeling, consistency training |
| **Reinforcement** | States, actions, rewards | Maximize cumulative reward | Game playing, robotics, RLHF |

> **Self-supervised learning is the engine of the modern AI era.** GPT is trained by hiding the
> next token and predicting it; BERT by masking tokens. The labels come free from the data,
> which is why these models can consume trillions of tokens.

### By task

| Task | Output | Loss | Example |
|------|--------|------|---------|
| **Regression** | $y \in \mathbb{R}$ | MSE, MAE, Huber | House price |
| **Binary classification** | $y \in \{0,1\}$ | BCE | Spam detection |
| **Multiclass** | $y\in\{1..K\}$, one label | Categorical CE | Digit recognition |
| **Multilabel** | subset of $K$ labels | Per-label BCE | Image tagging |
| **Ranking** | An ordering | Pairwise/listwise (NDCG) | Search results |
| **Generation** | A sequence / image | NLL, diffusion loss | LLMs, image models |
| **Structured prediction** | A structured object | Task-specific | Parsing, segmentation |

### By training mode

- **Batch/offline:** train once on all data, then deploy.
- **Online/incremental:** update as each example arrives.
- **Transfer learning:** start from a model trained on another task.
- **Fine-tuning:** continue training a pretrained model on your data.
- **Few-shot / zero-shot:** adapt with a handful of examples, or none (in-context learning).
- **Active learning:** the model chooses which examples to have labeled next.
- **Federated learning:** train across devices without centralizing data.

---

## 3. The bias–variance tradeoff

### The decomposition

For squared error, with true relationship $y = f(x) + \epsilon$ where
$\mathbb{E}[\epsilon]=0$, $\text{Var}(\epsilon)=\sigma^2$, the expected test error at a point
$x$ — averaged over possible training sets $\mathcal{D}$ — decomposes exactly:

$$\mathbb{E}_{\mathcal{D}}\big[(y - \hat f_\mathcal{D}(x))^2\big] = \underbrace{\big(\mathbb{E}_\mathcal{D}[\hat f_\mathcal{D}(x)] - f(x)\big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}_\mathcal{D}\big[(\hat f_\mathcal{D}(x) - \mathbb{E}_\mathcal{D}[\hat f_\mathcal{D}(x)])^2\big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible}}$$

| Term | Meaning | Cause | Symptom |
|------|---------|-------|---------|
| **Bias²** | Error from wrong assumptions — the model is systematically off | Model too simple | **Underfitting**: high train error AND high test error |
| **Variance** | Error from sensitivity to the particular training set | Model too complex | **Overfitting**: low train error, high test error |
| **Irreducible ($\sigma^2$)** | Noise inherent in the data | Missing features, measurement error, genuine randomness | Nothing fixes this |

### Diagnosing from learning curves

Plot training and validation error vs. training set size (and vs. epochs):

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Both errors high, converged together | **High bias** | Bigger model, more features, less regularization, train longer |
| Train error low, big gap to validation | **High variance** | More data, more regularization, simpler model, augmentation, early stopping |
| Both low, small gap | Good fit | Ship it |
| Train error > validation error | Bug, or heavy regularization/dropout at train time only | Check for leakage-in-reverse, check eval mode |
| Validation error much better than test | You overfit the validation set by tuning | Fresh test set |

### What reduces which

| Technique | Bias | Variance |
|-----------|------|----------|
| More training data | — | ↓↓ |
| More features / bigger model | ↓ | ↑ |
| Regularization ($\lambda\uparrow$) | ↑ | ↓ |
| **Bagging** (random forest) | — | ↓↓ |
| **Boosting** (XGBoost) | ↓↓ | ↑ (mildly) |
| Early stopping | ↑ | ↓ |
| Data augmentation | — | ↓ |
| Ensembling different models | ↓ | ↓ |

### Double descent (the modern wrinkle)

Classical theory predicts a U-shaped test error vs. capacity. Modern deep nets show
**double descent**: error rises to a peak at the interpolation threshold (where the model
has just enough capacity to fit the training data exactly), then *falls again* as capacity
grows further. Massively overparameterized networks generalize well — a phenomenon the
classical tradeoff doesn't explain, and an active research area. It's why "just make the
model bigger" often works in deep learning but not in classical ML.

---

## 4. Loss functions

The loss defines what "wrong" means. Choosing it is a modelling decision, not a detail.

### Regression losses

| Loss | Formula | Gradient behaviour | Use when |
|------|---------|-------------------|----------|
| **MSE / L2** | $\frac{1}{n}\sum(y_i-\hat y_i)^2$ | Grows with error; heavily penalizes outliers | Gaussian noise; you care about large errors |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same optimum as MSE | Reporting (same units as $y$) |
| **MAE / L1** | $\frac{1}{n}\sum\|y_i-\hat y_i\|$ | Constant magnitude; robust | Outliers present; predicts the **median** |
| **Huber** | see below | L2 near 0, L1 far away | Best of both |
| **Log-cosh** | $\sum\log(\cosh(y-\hat y))$ | Smooth Huber | Twice differentiable everywhere |
| **Quantile / pinball** | $\max(\tau e, (\tau-1)e)$ | Asymmetric | Prediction intervals; over/under-prediction cost differs |

**Huber loss:**

$$L_\delta(e) = \begin{cases} \frac{1}{2}e^2 & \text{if } |e|\le\delta \\ \delta(|e| - \frac{1}{2}\delta) & \text{otherwise}\end{cases}, \qquad e = y - \hat y$$

> **MSE predicts the conditional mean; MAE predicts the conditional median.** That's a
> different estimand, not just a different penalty. With skewed targets it changes your answer.

### Classification losses

**Binary cross-entropy (log loss):**

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\big]$$

**Categorical cross-entropy** ($K$ classes, one-hot $y$):

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}y_{ik}\log\hat p_{ik}$$

where $\hat{p}_{ik} = \text{softmax}(z_i)_k = \dfrac{e^{z_{ik}}}{\sum_j e^{z_{ij}}}$.

**The beautiful result:** for softmax + cross-entropy, the gradient with respect to the
*logits* is just:

$$\frac{\partial\mathcal{L}}{\partial z_k} = \hat p_k - y_k$$

Prediction minus truth. This clean form is why the pairing is universal — and why frameworks
fuse them into one op.

| Loss | Formula | Use |
|------|---------|-----|
| **Hinge** | $\max(0, 1 - y\cdot f(x))$, $y\in\{-1,1\}$ | SVM; margin maximization |
| **Focal** | $-\alpha(1-\hat p)^\gamma\log\hat p$ | Extreme class imbalance (object detection); down-weights easy examples |
| **Label-smoothed CE** | targets $(1-\epsilon)y + \epsilon/K$ | Prevents overconfidence, improves calibration |
| **KL divergence** | $\sum p\log(p/q)$ | Distillation, VAEs, policy constraints |
| **Contrastive / InfoNCE** | $-\log\frac{e^{s^+/\tau}}{\sum e^{s/\tau}}$ | Embedding learning, CLIP, retrieval |
| **Triplet** | $\max(0, d(a,p) - d(a,n) + m)$ | Face recognition, metric learning |

**Focal loss** in full: $\mathcal{L} = -\alpha_t(1-p_t)^\gamma\log(p_t)$. When an example is
already classified well ($p_t\to1$), the modulating factor $(1-p_t)^\gamma \to 0$ removes it
from the gradient, letting the hard examples dominate. $\gamma=2$ is standard.

### Choosing a loss: the decision rule

Ask: **what is the noise model, and what does an error actually cost me?** If false negatives
cost 10× false positives, encode that in the loss (class weights) rather than fixing it later
with a threshold — though threshold tuning is the cheaper first move.

---

## 5. Model validation

### The three-way split

| Split | Purpose | How often you may touch it |
|-------|---------|---------------------------|
| **Training** (60–80%) | Fit parameters | Constantly |
| **Validation / dev** (10–20%) | Tune hyperparameters, select models, early stopping | Many times |
| **Test** (10–20%) | Final unbiased estimate | **Once** |

Every time you make a decision based on the validation set, you leak a little information into
your model. After 100 experiments, your validation score is optimistically biased. The test set
is your only honest number — and it stops being honest the moment you tune against it.

### k-Fold cross-validation

Split the data into $k$ folds; train on $k-1$, validate on the held-out one; rotate; average.

$$\text{CV}_{(k)} = \frac{1}{k}\sum_{i=1}^{k}\text{Err}_i$$

| Variant | When to use |
|---------|-------------|
| **k-fold** ($k=5$ or 10) | Default |
| **Stratified k-fold** | Classification — preserves class proportions in each fold. **Use by default.** |
| **Leave-one-out (LOOCV)** | Tiny datasets; $k=n$; low bias, high variance, expensive |
| **Repeated k-fold** | More stable estimate; repeat with different seeds |
| **Group k-fold** | When rows share an entity (same user/patient/device) — keeps a group entirely within one fold |
| **TimeSeriesSplit** | Temporal data — always train on the past, validate on the future |
| **Nested CV** | Unbiased estimate *while* tuning hyperparameters: outer loop for evaluation, inner loop for tuning |

**Time series validation** (never shuffle):

```
Fold 1: train [1..100]  validate [101..120]
Fold 2: train [1..120]  validate [121..140]
Fold 3: train [1..140]  validate [141..160]
```

Optionally insert a **gap/embargo** between train and validation equal to your prediction
horizon, so information doesn't bleed across the boundary.

### Data leakage — the #1 cause of "great model, terrible production"

**Definition:** information available at training time that will not be available at prediction
time (or that encodes the target).

| Leakage type | Example | Fix |
|--------------|---------|-----|
| **Target leakage** | Feature `account_closed_date` when predicting churn | Ask: is this known *before* the event? |
| **Preprocessing leakage** | `StandardScaler` fit on the full dataset before splitting | Fit transformers **inside** the CV fold — use `Pipeline` |
| **Temporal leakage** | Random split on time-ordered data | Time-based split |
| **Group leakage** | Same patient's images in both train and test | `GroupKFold` |
| **Duplicate leakage** | Near-duplicate rows straddling the split | Deduplicate first |
| **Feature-selection leakage** | Selecting top-k features using all the data | Select inside the fold |

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ✅ scaler is re-fit on the training part of each fold only
pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
```

> **Rule of thumb:** if your first model is dramatically better than you expected, you have a
> leak. Go find it. This heuristic has never let me down.

---

## 6. Regularization

Any modification intended to reduce generalization error but not training error.

### L2 (Ridge / weight decay)

$$J(\theta) = \mathcal{L}(\theta) + \lambda\|\theta\|_2^2 = \mathcal{L}(\theta) + \lambda\sum_j\theta_j^2$$

Gradient: $\nabla J = \nabla\mathcal{L} + 2\lambda\theta$, so each step **shrinks** weights
multiplicatively toward zero — hence "weight decay". Shrinks correlated features *together*.
Never sets weights exactly to zero. Bayesian view: a Gaussian prior on the weights.

### L1 (Lasso)

$$J(\theta) = \mathcal{L}(\theta) + \lambda\|\theta\|_1 = \mathcal{L}(\theta) + \lambda\sum_j|\theta_j|$$

Produces **exactly zero** coefficients → automatic feature selection. Bayesian view: a Laplace prior.

**Why does L1 give sparsity and L2 not?** Geometric argument: minimizing loss subject to a
budget $\|\theta\|_1 \le t$ means finding where the elliptical loss contours first touch the
constraint region. The $L^1$ ball is a *diamond* with corners **on the axes** — a corner is
where some coordinates are exactly 0, and contours generically touch corners first. The $L^2$
ball is a smooth sphere with no corners, so the touch point almost never sits exactly on an axis.

Analytic view: the $L^1$ subgradient is $\lambda\,\text{sign}(\theta)$ — a constant push toward
zero that doesn't weaken as $\theta\to0$, so it can pin a coefficient at exactly 0. The $L^2$
gradient $2\lambda\theta$ vanishes as $\theta\to0$, so it never quite arrives.

### Elastic Net

$$J(\theta) = \mathcal{L}(\theta) + \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2$$

Sparsity from L1 plus stability from L2. Handles correlated feature *groups*, which lasso
handles badly (it arbitrarily picks one of a correlated set).

### Other regularizers

| Method | Mechanism | Where |
|--------|-----------|-------|
| **Early stopping** | Stop when validation error rises; equivalent to L2 for linear models | Everything |
| **Dropout** | Randomly zero units with probability $p$ during training; scale by $1/(1-p)$ | Neural nets |
| **Data augmentation** | Label-preserving transforms expand the effective dataset | Vision, audio, NLP |
| **Batch/Layer norm** | Stabilizes and has a mild regularizing effect | Deep nets |
| **Label smoothing** | Soften one-hot targets | Classification |
| **Max-norm / gradient clipping** | Constrain weight or gradient magnitude | RNNs, transformers |
| **Noise injection** | Add noise to inputs/weights/gradients | General |
| **Mixup / CutMix** | Train on convex combinations of examples and labels | Vision |
| **Ensembling** | Average multiple models | Everything |
| **Pruning tree depth / min samples** | Restrict capacity | Trees |

**Dropout details:** at training time, each unit is kept with probability $1-p$; at inference,
all units are used. To keep expected activations consistent, "inverted dropout" divides
activations by $(1-p)$ during training (what PyTorch does). Dropout approximates training an
exponential ensemble of subnetworks that share weights. Typical $p$: 0.5 for dense layers, 0.1
for transformers, rarely used in convolutions (batch norm handles it).

> **Forgetting `model.eval()`** leaves dropout and batch-norm in training mode at inference.
> This is the single most common PyTorch bug. Symptom: nondeterministic, worse-than-expected
> predictions.

---

## 7. Evaluation metrics

### The confusion matrix

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) — *Type II* |
| **Actually Negative** | False Positive (FP) — *Type I* | True Negative (TN) |

### Classification metrics

| Metric | Formula | Answers | Use when |
|--------|---------|---------|----------|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | Overall correctness | Balanced classes only |
| **Precision** (PPV) | $\frac{TP}{TP+FP}$ | "Of my positive predictions, how many were right?" | False positives are costly (spam) |
| **Recall** (Sensitivity, TPR) | $\frac{TP}{TP+FN}$ | "Of actual positives, how many did I catch?" | False negatives are costly (cancer) |
| **Specificity** (TNR) | $\frac{TN}{TN+FP}$ | True negative rate | Medical screening |
| **F1** | $2\cdot\frac{P\cdot R}{P+R}$ | Harmonic mean of P and R | Imbalanced; need both |
| **F-beta** | $(1+\beta^2)\frac{P\cdot R}{\beta^2 P + R}$ | Weighted; $\beta>1$ favours recall | Asymmetric costs |
| **Balanced accuracy** | $\frac{TPR + TNR}{2}$ | Class-averaged accuracy | Imbalanced |
| **MCC** | see below | Correlation between prediction and truth | Best single number for imbalanced binary |
| **Cohen's κ** | $\frac{p_o - p_e}{1 - p_e}$ | Agreement above chance | Annotator agreement |
| **Log loss** | $-\frac1n\sum[y\log\hat p + (1-y)\log(1-\hat p)]$ | Probability quality | You need calibrated probabilities |
| **Brier score** | $\frac1n\sum(\hat p_i - y_i)^2$ | Calibration + accuracy | Probabilistic forecasts |

**Matthews Correlation Coefficient** — robust to imbalance because it uses all four cells:

$$\text{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \in [-1, 1]$$

**Why the harmonic mean in F1?** It punishes imbalance between P and R. With P=1.0 and R=0.01,
the arithmetic mean is 0.505 (looks fine) but F1 = 0.0198 (correctly awful).

### Threshold-free metrics

**ROC curve:** TPR vs FPR as the decision threshold sweeps from 1 to 0.
**ROC-AUC** = probability that a randomly chosen positive is scored above a randomly chosen
negative. 0.5 = random; 1.0 = perfect.

**Precision–Recall curve** and **PR-AUC** (a.k.a. Average Precision).

> **Critical distinction:** ROC-AUC is *insensitive to class imbalance* because FPR has TN
> (a huge number) in its denominator — so a model can look great while being useless. With 0.1%
> positives, **use PR-AUC.** The PR baseline is the positive rate; the ROC baseline is always 0.5.

### Multiclass averaging

| Averaging | How | Use when |
|-----------|-----|----------|
| **Macro** | Mean of per-class metrics, unweighted | All classes matter equally (rare classes count fully) |
| **Micro** | Pool all TP/FP/FN globally (= accuracy for single-label) | Overall performance; large classes dominate |
| **Weighted** | Mean weighted by class support | Reporting on imbalanced data |

### Regression metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **MSE** | $\frac1n\sum(y-\hat y)^2$ | Penalizes big errors; units squared |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units as $y$ |
| **MAE** | $\frac1n\sum\|y-\hat y\|$ | Robust; interpretable |
| **MAPE** | $\frac{100}{n}\sum\left\|\frac{y-\hat y}{y}\right\|$ | Percent error; **breaks when $y\approx0$**; asymmetric |
| **SMAPE** | $\frac{100}{n}\sum\frac{\|y-\hat y\|}{(\|y\|+\|\hat y\|)/2}$ | Bounded alternative |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Fraction of variance explained; can be negative |
| **Adjusted R²** | $1-\frac{(1-R^2)(n-1)}{n-p-1}$ | Penalizes extra predictors |

$$R^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}$$

$R^2 = 0$ means "no better than predicting the mean"; negative means "worse than the mean".

### Calibration

A model is **calibrated** if, among predictions of 0.7, about 70% are actually positive.
High AUC does **not** imply calibration — ranking can be perfect while probabilities are
nonsense.

- **Diagnosis:** reliability diagram (predicted probability vs observed frequency, binned);
  **Expected Calibration Error** $\text{ECE} = \sum_b \frac{n_b}{n}|\text{acc}_b - \text{conf}_b|$.
- **Fixes:** **Platt scaling** (fit a logistic on the scores), **isotonic regression**
  (non-parametric, needs more data), **temperature scaling** (divide logits by a learned $T$ —
  the standard fix for neural nets).

Modern deep nets are systematically **overconfident**. If a downstream decision uses the
probability (expected-value calculations, thresholds tied to cost), you must calibrate.

### Choosing a metric

1. Start from the **business cost** of each error type.
2. Pick **one** primary metric to optimize (you can't optimize five).
3. Track guardrail metrics alongside (latency, fairness across subgroups, worst-group accuracy).
4. Always compare to a **baseline**: majority class, the mean, last-value-carried-forward, or
   the existing rule-based system.

---

## 8. Feature engineering & preprocessing

> "Applied machine learning is basically feature engineering." — Andrew Ng
> (Still true for tabular data; deep learning learns features instead for text/images/audio.)

### Numerical features

| Technique | Formula | When |
|-----------|---------|------|
| **Standardization (z-score)** | $z = \frac{x-\mu}{\sigma}$ | Default. Required for SVM, kNN, PCA, neural nets, any regularized linear model |
| **Min-max scaling** | $x' = \frac{x - \min}{\max-\min}$ | Bounded range needed; sensitive to outliers |
| **Robust scaling** | $\frac{x - \text{median}}{\text{IQR}}$ | Outliers present |
| **Log / log1p** | $\log(1+x)$ | Right-skewed, positive data (income, counts) |
| **Box–Cox / Yeo–Johnson** | power transform | Make data more Gaussian |
| **Binning / discretization** | quantile or fixed-width bins | Capture non-linearity in linear models |
| **Polynomial / interaction** | $x_1x_2$, $x^2$ | Explicit non-linearity |

**Tree-based models don't need scaling** — splits are invariant to monotone transforms.
Everything distance- or gradient-based does.

### Categorical features

| Encoding | How | Watch out for |
|----------|-----|---------------|
| **One-hot** | Binary column per category | Explodes with high cardinality |
| **Ordinal / label** | Map to integers | Only for genuinely ordered categories — otherwise implies false ordering |
| **Target / mean encoding** | Replace with the mean target for that category | **Leaks badly** — must use out-of-fold encoding + smoothing |
| **Frequency encoding** | Replace with the category's count | Cheap, surprisingly effective |
| **Hashing** | Hash to a fixed number of buckets | Collisions; but memory-bounded |
| **Embeddings** | Learn a dense vector per category | Best for high cardinality; needs a neural net |
| **CatBoost ordered TS** | Ordered target statistics | Built-in leak-free target encoding |

**Smoothed target encoding:**

$$\text{enc}(c) = \frac{n_c\cdot\bar y_c + m\cdot\bar y}{n_c + m}$$

where $n_c$ is the category's count, $\bar y_c$ its target mean, $\bar y$ the global mean, and
$m$ a smoothing strength. Rare categories get shrunk toward the global mean.

### Missing values

| Strategy | Notes |
|----------|-------|
| Drop rows/columns | Only if very few, or the column is mostly empty |
| Mean/median/mode imputation | Simple; distorts variance and correlations |
| Forward/backward fill | Time series |
| **Indicator column** | Add `is_missing` — **missingness is often predictive** |
| KNN imputation | Uses similar rows |
| Iterative (MICE) | Model each feature from the others |
| Native handling | XGBoost/LightGBM learn a default direction for missing values |

Understand the mechanism first: **MCAR** (missing completely at random — safe to impute),
**MAR** (missing at random given observed features), **MNAR** (missing *because of* the
unobserved value — e.g. high earners not reporting income; imputation will bias you).

### Text and datetime

**Text:** bag-of-words → TF-IDF → embeddings.

$$\text{TF-IDF}(t,d) = \underbrace{\frac{\text{count}(t,d)}{|d|}}_{\text{term frequency}}\times\underbrace{\log\frac{N}{1+|\{d: t\in d\}|}}_{\text{inverse document frequency}}$$

IDF down-weights words appearing in many documents ("the" carries no signal).

**Datetime:** never feed a raw timestamp. Extract hour, day-of-week, month, quarter,
is_weekend, is_holiday, days-since-event. Encode cyclical features so that hour 23 is adjacent
to hour 0:

$$x_{\sin} = \sin\left(\frac{2\pi t}{T}\right),\qquad x_{\cos} = \cos\left(\frac{2\pi t}{T}\right)$$

### Feature selection

| Family | Methods | Notes |
|--------|---------|-------|
| **Filter** | Correlation, chi², mutual information, variance threshold | Fast, model-agnostic |
| **Wrapper** | Forward/backward selection, RFE | Expensive, model-specific |
| **Embedded** | Lasso, tree importances | Free with training |
| **Permutation importance** | Shuffle a feature, measure metric drop | Model-agnostic, honest, but misleading with correlated features |
| **SHAP** | Shapley values from game theory | Best-in-class attribution; slower |

**Curse of dimensionality:** in high dimensions, data becomes sparse, all pairwise distances
converge to similar values (breaking kNN and clustering), and the volume needed to cover the
space grows exponentially. Rule of thumb: you want $n \gg d$; with $d$ near $n$, regularize
hard or reduce dimensions.

---

## 9. Data problems

### Class imbalance

| Approach | Method | Notes |
|----------|--------|-------|
| **Do nothing** | Just use the right metric + threshold | Often correct! Try this first |
| **Threshold tuning** | Move the decision threshold off 0.5 | Cheapest effective fix |
| **Class weights** | Weight the loss by $\frac{n}{K \cdot n_k}$ | `class_weight='balanced'`; no data changes |
| **Random undersampling** | Drop majority examples | Loses information |
| **Random oversampling** | Duplicate minority examples | Overfits duplicates |
| **SMOTE** | Synthesize minority points along lines between neighbours | Can create nonsense in high dimensions; apply **inside** the CV fold |
| **Focal loss** | Down-weight easy examples | Extreme imbalance |
| **Anomaly detection framing** | Treat as one-class | When positives < 0.1% |

> **Never resample the validation or test set.** Your evaluation must reflect the real,
> imbalanced world. Resample only the training fold.

### Outliers

Detect: z-score $|z|>3$, IQR rule, isolation forest, DBSCAN noise points, Mahalanobis distance.
Handle: remove (only if a genuine error), winsorize (clip to percentiles), transform (log),
or use robust models/losses (Huber, tree-based).

**Ask first: is it an error or a real extreme event?** In fraud detection the outliers *are*
the signal.

### Distribution shift

| Type | What changes | Example |
|------|-------------|---------|
| **Covariate shift** | $P(x)$ changes, $P(y\mid x)$ stable | New user demographics |
| **Label / prior shift** | $P(y)$ changes | Fraud rate rises seasonally |
| **Concept drift** | $P(y\mid x)$ changes | Post-pandemic spending behaviour |
| **Domain shift** | Different source entirely | Trained on stock photos, deployed on phone cameras |

Detection: KS test / PSI on feature distributions, monitoring prediction distributions, and
(when labels arrive) monitoring live metrics. See [Module 09](09_mlops_and_deployment.md).

---

## 10. Hyperparameter tuning

| Method | How | When |
|--------|-----|------|
| **Manual** | Educated guesses | Always the starting point — know your defaults |
| **Grid search** | Exhaustive over a grid | Few hyperparameters (≤3), cheap models |
| **Random search** | Sample randomly from distributions | **Better than grid** — Bergstra & Bengio (2012) showed random search finds better configs in the same budget, because only a few hyperparameters matter and random search tries more distinct values for each |
| **Bayesian optimization** | Fit a surrogate (Gaussian process / TPE), sample where expected improvement is highest | Expensive training runs; Optuna, Hyperopt |
| **Hyperband / ASHA** | Successive halving: start many configs, kill the bad ones early | Large search spaces, deep learning |
| **Population-based training** | Evolutionary; mutates hyperparameters during training | Large-scale RL/DL |

**Priority order (what actually matters):**

| Model | Tune first | Then | Rarely matters |
|-------|-----------|------|----------------|
| **Neural nets** | Learning rate | Batch size, architecture width/depth, weight decay | Optimizer $\epsilon$, $\beta_2$ |
| **Gradient boosting** | `learning_rate` + `n_estimators` (together) | `max_depth`/`num_leaves`, `min_child_weight`, `subsample`, `colsample` | `gamma`, `lambda` |
| **Random forest** | `max_features` | `min_samples_leaf`, `max_depth` | `n_estimators` (more is just slower, never worse) |
| **SVM** | `C`, `gamma` | kernel | — |

**Learning rate is worth more tuning attention than everything else combined** in deep learning.
Search it on a log scale.

---

## 11. The ML workflow

```
1. FRAME       What decision does this support? What does success mean numerically?
               What's the baseline? Is ML even the right tool?
2. DATA        Collect → understand (EDA) → clean → split (BEFORE any transform)
3. BASELINE    Majority class / mean / simple rule / logistic regression.
               Anything you build later must beat this or it's not worth its complexity.
4. FEATURES    Engineer, encode, scale — all inside a Pipeline
5. MODEL       Start simple. Add complexity only when validation says it helps.
6. VALIDATE    Correct CV scheme for your data structure. Hunt for leakage.
7. TUNE        Random/Bayesian search on the validation set
8. ANALYZE     Error analysis: WHERE does it fail? Which subgroups? Look at 50 errors by hand.
9. TEST        Touch the test set once. Report honestly with confidence intervals.
10. DEPLOY     Serve, monitor, detect drift, plan retraining  → Module 09
11. ITERATE    Production feedback is the most valuable data you will ever get
```

### Error analysis: the highest-leverage activity

Sort validation errors by loss and **read the worst 50 by hand.** Categorize them. You will
almost always find that 60% of your error comes from one fixable cause: mislabeled data, a
specific subgroup, a preprocessing bug, or a genuinely ambiguous class boundary. No amount of
hyperparameter tuning finds that.

### Things practitioners actually get wrong

1. Optimizing a metric that doesn't reflect the decision being made.
2. Not establishing a baseline, so "0.85 AUC" is meaningless.
3. Leakage (again — it's that common).
4. Tuning on the test set.
5. Ignoring class imbalance in metrics but "fixing" it with SMOTE everywhere.
6. Deploying without monitoring.
7. Reaching for deep learning on 5,000 rows of tabular data. (Use gradient boosting.)
8. Reporting a single number without any measure of variance.

---

## Self-check

1. Write the ERM objective and explain each term.
2. Decompose test error into three parts. Which does bagging reduce? Boosting?
3. Give the geometric reason L1 produces sparsity.
4. When is ROC-AUC misleading? What replaces it?
5. Name five distinct ways data leakage enters a pipeline.
6. Your model has AUC 0.92 but its probabilities are unusable for expected-value decisions. What's wrong, and how do you fix it?
7. Why is random search better than grid search at equal budget?
8. Why must SMOTE be applied inside the CV fold?

---

**Next:** [03 — Supervised Learning →](03_supervised_learning.md)
