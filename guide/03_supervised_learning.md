# 03 — Supervised Learning Algorithms

> Every classical algorithm: the intuition, the mathematics, the derivation, the
> hyperparameters that matter, and the failure modes. Each links to a runnable script in
> this repo.

---

## Table of contents

1. [Linear Regression](#1-linear-regression)
2. [Logistic Regression](#2-logistic-regression)
3. [k-Nearest Neighbours (kNN)](#3-k-nearest-neighbours-knn)
4. [Naive Bayes](#4-naive-bayes)
5. [Support Vector Machines (SVM)](#5-support-vector-machines-svm)
6. [Decision Trees](#6-decision-trees)
7. [Bagging and Random Forests](#7-bagging-and-random-forests)
8. [Boosting](#8-boosting)
9. [Discriminant Analysis (LDA/QDA)](#9-discriminant-analysis-ldaqda)
10. [Generalized Linear Models](#10-generalized-linear-models)
11. [Model comparison cheat sheet](#11-model-comparison-cheat-sheet)
12. [Time series](#12-time-series)

---

## 1. Linear Regression

📄 `supervised/linear_regression_example.py`

### Intuition
Fit the best straight line (or hyperplane) through the data.

### Model

$$\hat y = w_0 + w_1x_1 + \cdots + w_dx_d = \mathbf{w}^\top\mathbf{x} + b$$

In matrix form, absorbing the bias by prepending a column of ones to $X$:

$$\hat{\mathbf{y}} = X\mathbf{w}, \qquad X\in\mathbb{R}^{n\times(d+1)}$$

### Objective

Minimize the residual sum of squares:

$$J(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 = (\mathbf{y}-X\mathbf{w})^\top(\mathbf{y}-X\mathbf{w})$$

### Derivation of the closed-form solution

Expand:

$$J(\mathbf{w}) = \mathbf{y}^\top\mathbf{y} - 2\mathbf{w}^\top X^\top\mathbf{y} + \mathbf{w}^\top X^\top X\mathbf{w}$$

Differentiate using the matrix calculus rules from [Module 01 §1.10](01_math_foundations.md#110-matrix-calculus-the-gradient-rules-youll-actually-use):

$$\nabla_\mathbf{w}J = -2X^\top\mathbf{y} + 2X^\top X\mathbf{w}$$

Set to zero → the **normal equations**:

$$X^\top X\mathbf{w} = X^\top\mathbf{y} \quad\Longrightarrow\quad \boxed{\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}}$$

Since $J$ is convex ($\nabla^2 J = 2X^\top X \succeq 0$), this stationary point is the global
minimum. ∎

**Geometric meaning:** $\hat{\mathbf{y}} = X(X^\top X)^{-1}X^\top\mathbf{y} = H\mathbf{y}$,
where $H$ is the **hat matrix** — an orthogonal projection of $\mathbf{y}$ onto the column
space of $X$. Least squares finds the closest point in the span of your features. The residual
is orthogonal to every feature: $X^\top(\mathbf{y}-\hat{\mathbf{y}}) = 0$.

### When the closed form fails

$X^\top X$ is singular when features are linearly dependent (**multicollinearity**) or when
$d > n$. Then infinitely many solutions exist. Remedies: drop/combine features, use the
pseudo-inverse, or **regularize**.

**Cost:** $O(nd^2 + d^3)$. For large $d$, use gradient descent ($O(nd)$ per step) instead.

### Regularized variants

**Ridge (L2):**

$$J(\mathbf{w}) = \|\mathbf{y}-X\mathbf{w}\|_2^2 + \lambda\|\mathbf{w}\|_2^2 \quad\Longrightarrow\quad \hat{\mathbf{w}}_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}$$

The $\lambda I$ term makes the matrix **always invertible** (it shifts every eigenvalue up by
$\lambda$) — ridge solves multicollinearity structurally, not just statistically.

**Lasso (L1):** $J = \|\mathbf{y}-X\mathbf{w}\|_2^2 + \lambda\|\mathbf{w}\|_1$. No closed
form (the $L^1$ term isn't differentiable at 0); solved by coordinate descent using the
**soft-thresholding** operator:

$$S(z,\gamma) = \text{sign}(z)\max(|z|-\gamma, 0)$$

**Elastic Net:** both penalties.

### Assumptions (the classical five)
1. **Linearity** — the relationship is linear in the *parameters*.
2. **Independence** — residuals are uncorrelated (violated by time series).
3. **Homoscedasticity** — constant residual variance (check: residuals vs fitted plot).
4. **Normality of residuals** — needed for p-values/CIs, not for the point estimate.
5. **No perfect multicollinearity** — check with **VIF**: $\text{VIF}_j = \frac{1}{1-R_j^2}$;
   $>10$ signals a problem.

### Diagnostics
Residual plot (should be structureless noise), Q-Q plot (normality), Cook's distance
(influential points), VIF (collinearity).

### In practice
- Interpretation: $w_j$ = expected change in $y$ per unit change in $x_j$, **holding all other
  features fixed** — that caveat is doing enormous work and is often unrealistic.
- Always scale features before regularizing, or the penalty is applied unevenly.
- Don't regularize the intercept.

---

## 2. Logistic Regression

📄 `supervised/logistic_regression_example.py`

### Intuition
Linear regression's output is unbounded, but probabilities live in $[0,1]$. Squash the linear
output through a sigmoid.

### Model

$$P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}+b) = \frac{1}{1+e^{-(\mathbf{w}^\top\mathbf{x}+b)}}$$

**Sigmoid (logistic) function:**

$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma(z)\in(0,1), \qquad \sigma'(z) = \sigma(z)(1-\sigma(z))$$

That derivative identity is worth memorizing — it makes the gradients collapse beautifully.

### The logit / odds view

$$\text{odds} = \frac{p}{1-p}, \qquad \text{logit}(p) = \log\frac{p}{1-p} = \mathbf{w}^\top\mathbf{x}+b$$

**Logistic regression is a linear model in log-odds space.** Interpretation: a one-unit
increase in $x_j$ multiplies the odds by $e^{w_j}$ (the **odds ratio**).

### Objective — derived from MLE

Each label is Bernoulli: $P(y\mid\mathbf{x}) = p^{y}(1-p)^{1-y}$. The log-likelihood is:

$$\log L = \sum_{i=1}^n\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$

Negating gives **binary cross-entropy** — so logistic regression's loss isn't a choice, it's a
consequence of assuming Bernoulli labels:

$$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i\log\hat p_i + (1-y_i)\log(1-\hat p_i)\big]$$

### Gradient derivation

With $z_i = \mathbf{w}^\top\mathbf{x}_i$ and $\hat p_i = \sigma(z_i)$:

$$\frac{\partial J}{\partial \hat p_i} = -\frac{y_i}{\hat p_i} + \frac{1-y_i}{1-\hat p_i} = \frac{\hat p_i - y_i}{\hat p_i(1-\hat p_i)}$$

$$\frac{\partial \hat p_i}{\partial z_i} = \hat p_i(1-\hat p_i)$$

Multiplying, the awkward denominator **cancels exactly**:

$$\frac{\partial J}{\partial z_i} = \hat p_i - y_i \quad\Longrightarrow\quad \boxed{\nabla_\mathbf{w}J = \frac{1}{n}X^\top(\hat{\mathbf{p}} - \mathbf{y})}$$

Identical in form to linear regression's gradient. No closed-form solution exists, so we use
gradient descent, Newton's method (**IRLS**), or L-BFGS.

> **Why not use MSE with a sigmoid?** MSE + sigmoid is non-convex in $\mathbf{w}$ and its
> gradient contains a $\sigma'(z)$ factor that vanishes when the model is confidently wrong —
> learning stalls exactly when it most needs to move. Cross-entropy's gradient
> $(\hat p - y)$ is *largest* when confidently wrong. This is the saturation problem.

### Multiclass: softmax regression

$$P(y=k\mid\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top\mathbf{x}}}{\sum_{j=1}^{K}e^{\mathbf{w}_j^\top\mathbf{x}}}$$

Loss: categorical cross-entropy. Gradient w.r.t. logits: $\hat p_k - y_k$ again.

Alternatives for multiclass with binary learners: **one-vs-rest** ($K$ classifiers) and
**one-vs-one** ($\binom{K}{2}$ classifiers).

### In practice
- Regularization is on by default in `sklearn` (`C = 1/λ`, so **smaller `C` = stronger
  regularization**). This trips people up constantly.
- Scale your features.
- **Perfect separation** (a feature that perfectly splits classes) sends weights to infinity;
  regularization is what keeps it finite.
- Naturally well-calibrated — one of its underrated strengths.
- Still the right first model for most binary classification problems: fast, interpretable, a
  strong baseline, and it tells you immediately whether the problem is linearly separable.

---

## 3. k-Nearest Neighbours (kNN)

📄 `supervised/knn_classifier_example.py`

### Intuition
To classify a new point, look at the $k$ closest training points and take a vote.

### Algorithm

```
predict(x):
    distances = [d(x, x_i) for x_i in training_set]
    neighbors = k points with smallest distance
    classification: return majority vote of neighbors' labels
    regression:     return mean of neighbors' values
```

No training phase — a **lazy / instance-based** learner. It memorizes the data and defers all
computation to prediction time. It's also **non-parametric**: the number of "parameters" grows
with the data.

### Distance metrics

| Metric | Formula |
|--------|---------|
| **Euclidean** ($L^2$) | $d = \sqrt{\sum_j(x_j - x'_j)^2}$ |
| **Manhattan** ($L^1$) | $d = \sum_j\|x_j - x'_j\|$ |
| **Minkowski** ($L^p$) | $d = \left(\sum_j\|x_j-x'_j\|^p\right)^{1/p}$ |
| **Cosine** | $1 - \frac{\mathbf{x}^\top\mathbf{x}'}{\\|\mathbf{x}\\|\\|\mathbf{x}'\\|}$ |
| **Hamming** | count of differing positions (categorical) |
| **Mahalanobis** | $\sqrt{(\mathbf{x}-\mathbf{x}')^\top\Sigma^{-1}(\mathbf{x}-\mathbf{x}')}$ (accounts for correlation) |

### Choosing k

- Small $k$ → low bias, high variance, jagged decision boundary, sensitive to noise.
- Large $k$ → high bias, low variance, smooth boundary; $k=n$ predicts the global majority.
- Use odd $k$ for binary classification to avoid ties. Tune by CV. Rule of thumb: $k\approx\sqrt{n}$.

**Distance weighting:** weight neighbours by $1/d$ so closer points count more.

### Complexity & scaling

Brute force: $O(nd)$ per query — unusable for large $n$. Accelerate with **KD-trees**
($O(\log n)$ for $d \lesssim 20$), **ball trees**, or approximate methods (**HNSW**, **IVF**,
**LSH**) — the last of which is exactly what vector databases in RAG systems use.

### Critical requirements
- **Scaling is mandatory.** An unscaled feature with a large range dominates the distance.
- **Curse of dimensionality:** in high $d$, all points become roughly equidistant and the
  notion of "nearest" loses meaning. Reduce dimensions (PCA) first.
- Struggles with imbalanced classes (the majority class wins votes by sheer density).

---

## 4. Naive Bayes

### Intuition
Apply Bayes' theorem, and "naively" assume all features are conditionally independent given
the class. The assumption is almost always false, and it works anyway.

### Derivation

$$P(y\mid x_1,\dots,x_d) = \frac{P(x_1,\dots,x_d\mid y)P(y)}{P(x_1,\dots,x_d)}$$

The naive conditional-independence assumption factorizes the likelihood:

$$P(x_1,\dots,x_d\mid y) = \prod_{j=1}^{d}P(x_j\mid y)$$

The denominator is constant across classes, so:

$$\hat y = \arg\max_{k}\; P(y=k)\prod_{j=1}^{d}P(x_j\mid y=k)$$

In practice work in log space to avoid underflow:

$$\hat y = \arg\max_k\left[\log P(y=k) + \sum_{j=1}^{d}\log P(x_j\mid y=k)\right]$$

### Variants

| Variant | Likelihood $P(x_j\mid y)$ | Use |
|---------|--------------------------|-----|
| **Gaussian NB** | $\frac{1}{\sqrt{2\pi\sigma_{jk}^2}}\exp\left(-\frac{(x_j-\mu_{jk})^2}{2\sigma_{jk}^2}\right)$ | Continuous features |
| **Multinomial NB** | $\frac{\text{count}(x_j, k) + \alpha}{\text{count}(k) + \alpha d}$ | Word counts — text classification |
| **Bernoulli NB** | Binary presence/absence | Binary features, short text |
| **Complement NB** | Uses complement class statistics | Imbalanced text |

**Laplace (additive) smoothing** — the $\alpha$ above (typically 1). Without it, a single
word never seen with a class gives $P=0$, which zeroes the entire product regardless of all
other evidence.

### Why it works despite a false assumption
The independence assumption ruins the *probability estimates* (they're wildly overconfident)
but often preserves the *argmax* — the ranking of classes survives. If you only need the
predicted class, you're fine; if you need calibrated probabilities, you're not.

### In practice
Extremely fast ($O(nd)$ training, single pass), works with tiny datasets, excellent for text
(spam filtering, document classification), and a great baseline. Poor at capturing feature
interactions by construction.

---

## 5. Support Vector Machines (SVM)

📄 `supervised/svm_classifier_example.py`

### Intuition
Of all hyperplanes separating two classes, choose the one with the **widest margin** —
the largest buffer between the classes. Intuitively, the most robust boundary.

### Hard-margin formulation

With labels $y_i\in\{-1,+1\}$, the hyperplane is $\mathbf{w}^\top\mathbf{x}+b = 0$. The
distance from a point to it is $\frac{|\mathbf{w}^\top\mathbf{x}+b|}{\|\mathbf{w}\|}$. Fixing
the scale so the closest points satisfy $|\mathbf{w}^\top\mathbf{x}+b| = 1$, the margin width
is $\frac{2}{\|\mathbf{w}\|}$.

Maximizing $\frac{2}{\|\mathbf{w}\|}$ = minimizing $\frac{1}{2}\|\mathbf{w}\|^2$:

$$\min_{\mathbf{w},b}\;\frac{1}{2}\|\mathbf{w}\|^2 \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i+b)\ge 1\;\;\forall i$$

A convex quadratic program — a unique global optimum.

### Soft margin (real data isn't separable)

Introduce **slack variables** $\xi_i \ge 0$ permitting violations:

$$\min_{\mathbf{w},b,\xi}\;\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i \quad\text{s.t.}\quad y_i(\mathbf{w}^\top\mathbf{x}_i+b)\ge 1-\xi_i,\;\;\xi_i\ge0$$

Equivalently, in unconstrained **hinge loss** form:

$$\min_\mathbf{w}\;\underbrace{\frac{1}{2}\|\mathbf{w}\|^2}_{\text{margin}} + C\sum_{i=1}^{n}\underbrace{\max\big(0,\,1-y_i(\mathbf{w}^\top\mathbf{x}_i+b)\big)}_{\text{hinge loss}}$$

**$C$ is the regularization knob:** small $C$ → wide margin, more violations tolerated
(more regularization, higher bias). Large $C$ → narrow margin, few violations (risk of
overfitting). Note it's the *inverse* of $\lambda$.

### The dual and the kernel trick

Applying Lagrange multipliers ([Module 01 §2.6](01_math_foundations.md#26-lagrange-multipliers-constrained-optimization))
gives the dual problem:

$$\max_{\boldsymbol\alpha}\;\sum_{i=1}^n\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_iy_j\,\mathbf{x}_i^\top\mathbf{x}_j \quad\text{s.t.}\quad 0\le\alpha_i\le C,\;\;\sum_i\alpha_iy_i=0$$

Two crucial observations:

1. **Complementary slackness** implies $\alpha_i = 0$ for every point outside the margin.
   Only points *on or inside* the margin have $\alpha_i > 0$ — these are the **support
   vectors**, and they alone define the boundary. You could delete every other training point
   and get the identical model.
2. The data appears **only as inner products** $\mathbf{x}_i^\top\mathbf{x}_j$.

Observation 2 enables the **kernel trick**: replace the inner product with a kernel function
$K(\mathbf{x}_i,\mathbf{x}_j) = \phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$, which computes
similarity in a high-dimensional feature space **without ever computing $\phi$**.

### Kernels

| Kernel | $K(\mathbf{x},\mathbf{x}')$ | Notes |
|--------|------------------------------|-------|
| **Linear** | $\mathbf{x}^\top\mathbf{x}'$ | High-dimensional sparse data (text) |
| **Polynomial** | $(\gamma\,\mathbf{x}^\top\mathbf{x}' + r)^p$ | Feature interactions up to degree $p$ |
| **RBF / Gaussian** | $\exp(-\gamma\\|\mathbf{x}-\mathbf{x}'\\|^2)$ | **Default.** Infinite-dimensional feature space |
| **Sigmoid** | $\tanh(\gamma\,\mathbf{x}^\top\mathbf{x}'+r)$ | Rarely used |

**Mercer's condition:** a valid kernel's Gram matrix $K_{ij}=K(\mathbf{x}_i,\mathbf{x}_j)$ must
be symmetric positive semi-definite — that guarantees a corresponding $\phi$ exists.

**$\gamma$ in the RBF kernel** controls the reach of a single training example: large $\gamma$
= narrow influence = wiggly boundary = overfitting; small $\gamma$ = broad influence = smooth
boundary = underfitting. `gamma='scale'` (= $1/(d\cdot\text{Var}(X))$) is a sound default.

### Prediction

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i\in SV}\alpha_iy_iK(\mathbf{x}_i,\mathbf{x}) + b\right)$$

### In practice
- **Scaling is mandatory** (the kernel is distance-based).
- Training is $O(n^2)$ to $O(n^3)$ — impractical beyond ~100k rows. Use `LinearSVC` (liblinear)
  or SGD-based linear models at scale.
- Tune `C` and `gamma` together on a log grid — they interact strongly.
- No native probabilities; `probability=True` runs internal Platt scaling with cross-validation
  (slow, and it can disagree with `predict`).
- **SVR** (support vector regression) uses an $\epsilon$-insensitive tube: no penalty for
  errors within $\epsilon$.

---

## 6. Decision Trees

📄 `supervised/decision_tree_classifier_example.py`

### Intuition
A flowchart of yes/no questions. Each internal node tests one feature against a threshold;
each leaf gives a prediction.

### Growing a tree (CART, recursive greedy partitioning)

```
build(node, data):
    if stopping_criterion(data): make leaf with majority class / mean value; return
    best = argmax over (feature, threshold) of impurity_decrease(split)
    left, right = partition(data, best)
    build(left);  build(right)
```

At each node, evaluate **every** feature and **every** candidate threshold, pick the split
that most reduces impurity. Greedy and locally optimal — finding the globally optimal tree is
NP-complete.

### Impurity measures (classification)

Let $p_k$ be the proportion of class $k$ at a node.

**Gini impurity** — probability of misclassifying a randomly chosen element if labeled by the
node's class distribution:

$$G = \sum_{k=1}^{K}p_k(1-p_k) = 1 - \sum_{k=1}^{K}p_k^2$$

**Entropy:**

$$H = -\sum_{k=1}^{K}p_k\log_2 p_k$$

**Information gain** — the impurity reduction from a split:

$$IG = H(\text{parent}) - \sum_{c\in\text{children}}\frac{n_c}{n}H(c)$$

Both peak at a 50/50 split (Gini 0.5, entropy 1.0 bit) and are 0 for a pure node. Gini is
slightly cheaper (no logarithm) and is `sklearn`'s default; results are nearly always the same.

**Regression** uses variance reduction (equivalently, MSE):

$$\text{MSE}_{\text{node}} = \frac{1}{n}\sum_{i\in\text{node}}(y_i - \bar y_{\text{node}})^2$$

### Stopping and pruning

| Hyperparameter | Effect |
|----------------|--------|
| `max_depth` | Hard cap on depth — **the main overfitting control** |
| `min_samples_split` | Don't split nodes smaller than this |
| `min_samples_leaf` | Every leaf must have at least this many samples |
| `max_features` | Consider only a random subset of features per split |
| `min_impurity_decrease` | Require a minimum gain to split |
| `ccp_alpha` | **Cost-complexity pruning** strength |

**Cost-complexity (weakest-link) pruning** — grow fully, then prune back to minimize:

$$R_\alpha(T) = R(T) + \alpha|T|$$

where $R(T)$ is the tree's error and $|T|$ its number of leaves. Sweep $\alpha$ and pick by CV.

### Strengths and weaknesses

| ✅ | ❌ |
|---|---|
| Interpretable (you can draw it) | **High variance** — small data changes → totally different tree |
| No scaling needed | Overfits readily without constraints |
| Handles mixed feature types | Axis-aligned splits only — struggles with diagonal boundaries |
| Handles non-linearity and interactions natively | Cannot extrapolate beyond the training range |
| Fast prediction, $O(\text{depth})$ | Biased toward high-cardinality features |
| Missing values handleable via surrogate splits | Unstable → which is exactly why we ensemble them |

**Feature importance** in trees is the total impurity decrease attributable to each feature,
weighted by the samples reaching those nodes. Be warned: this measure is **biased toward
high-cardinality and continuous features**. Prefer permutation importance or SHAP.

---

## 7. Bagging and Random Forests

📄 `supervised/random_forest_classifier_example.py`

### Bagging (Bootstrap AGGregatING)

1. Draw $B$ bootstrap samples (sample $n$ rows **with replacement**) from the training set.
2. Train an independent model on each.
3. Aggregate: majority vote (classification) or mean (regression).

**Why it works — the variance math.** For $B$ i.i.d. predictors each with variance $\sigma^2$,
the average has variance $\sigma^2/B$. Bootstrap samples aren't fully independent, so with
pairwise correlation $\rho$:

$$\text{Var}(\bar f) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

The second term $\to 0$ as $B\to\infty$, but the first term $\rho\sigma^2$ is a **floor**.
**To improve further you must reduce $\rho$ — decorrelate the trees.** That single insight is
the whole idea behind random forests.

Bagging reduces variance and leaves bias roughly unchanged — so use it on **low-bias,
high-variance** base learners: fully grown trees.

### Random Forest = Bagging + feature subsampling

At **each split** (not each tree), consider only a random subset of `max_features` features.
This prevents every tree from keying on the same dominant feature, lowering $\rho$.

Defaults: $\sqrt{d}$ features for classification, $d/3$ for regression.

### Out-of-Bag (OOB) estimation

Each bootstrap sample omits roughly

$$\lim_{n\to\infty}\left(1-\frac{1}{n}\right)^n = \frac{1}{e}\approx 36.8\%$$

of the data. Evaluating each tree on its own out-of-bag rows gives a **free validation
estimate** with no separate holdout (`oob_score=True`).

### Hyperparameters

| Parameter | Guidance |
|-----------|----------|
| `n_estimators` | More is always better for accuracy (it plateaus), just slower. 100–1000. **Never overfits by increasing this.** |
| `max_features` | The key tuning knob. Lower = more decorrelation |
| `max_depth` / `min_samples_leaf` | Usually leave unrestricted; the ensemble handles variance |
| `n_jobs=-1` | Trees are independent → embarrassingly parallel |

### Extremely Randomized Trees (Extra Trees)
Also randomizes the *threshold* — picks split points at random instead of optimizing them.
Even more decorrelation, more bias, less variance, and faster to train.

---

## 8. Boosting

📄 `python/10_xgboost_lightgbm.py`

### Intuition
Instead of averaging independent models (bagging), build models **sequentially**, each one
fixing the previous ensemble's mistakes. Bagging reduces variance; **boosting reduces bias**.

### AdaBoost

Maintain a weight per training example; increase the weights of misclassified examples so the
next learner focuses there.

```
w_i = 1/n for all i
for m = 1..M:
    fit weak learner h_m using weights w
    err_m = Σ w_i·1[h_m(x_i) ≠ y_i] / Σ w_i
    α_m   = ½·ln((1 - err_m)/err_m)
    w_i  ← w_i·exp(-α_m·y_i·h_m(x_i))     # up-weight the errors
    normalize w
final: H(x) = sign(Σ α_m·h_m(x))
```

$\alpha_m$ is the learner's vote weight: a learner with 50% error gets $\alpha=0$ (no vote); a
better-than-chance one gets positive weight. AdaBoost is provably equivalent to forward
stagewise additive modelling with an **exponential loss** $e^{-yf(x)}$.

### Gradient Boosting

Generalize to any differentiable loss: each new tree fits the **negative gradient** of the loss
with respect to the current predictions (the "pseudo-residuals").

```
F_0(x) = argmin_γ Σ L(y_i, γ)              # e.g. the mean for MSE
for m = 1..M:
    r_im = -[∂L(y_i, F(x_i)) / ∂F(x_i)]    evaluated at F = F_{m-1}
    fit regression tree h_m to the targets r_im
    γ_m = argmin_γ Σ L(y_i, F_{m-1}(x_i) + γ·h_m(x_i))
    F_m(x) = F_{m-1}(x) + ν·γ_m·h_m(x)     # ν = learning rate (shrinkage)
```

For squared loss, the negative gradient is exactly the residual $y_i - F(x_i)$ — so "fit the
residuals" is the special case, and gradient boosting is the generalization to any loss.

**This is gradient descent in function space:** each tree is a step in the direction that most
decreases the loss, and $\nu$ is the step size.

### XGBoost — the second-order refinement

XGBoost uses a second-order Taylor expansion of the loss and adds explicit regularization:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n}\left[g_i f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i f_t^2(\mathbf{x}_i)\right] + \Omega(f_t)$$

with $g_i = \partial_{\hat y}L$, $h_i = \partial^2_{\hat y}L$, and

$$\Omega(f) = \gamma T + \tfrac{1}{2}\lambda\sum_{j=1}^{T}w_j^2$$

($T$ = number of leaves, $w_j$ = leaf weights). Solving for the optimal leaf weight gives:

$$w_j^* = -\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i + \lambda}$$

and the **gain** used to score a candidate split:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

The $-\gamma$ term means a split must earn its own complexity cost — built-in pruning.

### The gradient boosting family

| Library | Distinguishing feature |
|---------|----------------------|
| **XGBoost** | Second-order gradients, regularized objective, sparsity-aware, histogram splits |
| **LightGBM** | **Leaf-wise** growth (vs level-wise) + GOSS + EFB → much faster on large data; can overfit small data |
| **CatBoost** | Ordered boosting (removes target-leakage bias) + native categorical handling; best defaults |
| **sklearn HistGradientBoosting** | LightGBM-style, no extra dependency |

### Hyperparameters that matter

| Parameter | Typical | Effect |
|-----------|---------|--------|
| `learning_rate` (ν) | 0.01–0.1 | Lower = better generalization, needs more trees |
| `n_estimators` | 100–5000 | Set high + **early stopping** on a validation set |
| `max_depth` | 3–8 | Boosted trees should be **shallow** (weak learners) |
| `num_leaves` (LGBM) | < $2^{\text{max\_depth}}$ | The main capacity knob for leaf-wise growth |
| `subsample` | 0.6–1.0 | Row sampling → stochastic gradient boosting |
| `colsample_bytree` | 0.6–1.0 | Column sampling |
| `min_child_weight` | 1–100 | Minimum hessian sum per leaf; higher = more conservative |
| `reg_lambda`, `reg_alpha` | 0–10 | L2 / L1 on leaf weights |

**The core tradeoff:** `learning_rate` × `n_estimators` ≈ constant. Halve the rate, double the
trees. Always use early stopping rather than guessing `n_estimators`.

> **Unlike random forests, boosting CAN overfit with too many trees.** Early stopping isn't
> optional.

### Why gradient boosting dominates tabular ML
It handles mixed types, missing values, non-linearity, and interactions with essentially no
preprocessing, and it consistently beats neural networks on tabular data at realistic dataset
sizes. **On a new tabular problem, your strong baseline is LightGBM/XGBoost with early
stopping.** Deep learning is not the answer there.

### Bagging vs Boosting

| | Bagging (RF) | Boosting (XGB) |
|---|---|---|
| Training | Parallel, independent | Sequential, dependent |
| Base learners | Deep (low-bias) trees | Shallow (high-bias) trees |
| Primarily reduces | **Variance** | **Bias** |
| Overfits with more estimators? | No | **Yes** |
| Hyperparameter sensitivity | Low | High |
| Typical accuracy | Very good | **Usually best** |
| Robust to noisy labels | More | Less (chases the noise) |

---

## 9. Discriminant Analysis (LDA/QDA)

**Generative** classifiers: model $P(\mathbf{x}\mid y)$ and $P(y)$, then apply Bayes' rule —
in contrast to *discriminative* models (logistic regression) that model $P(y\mid\mathbf{x})$
directly.

Assume each class is Gaussian: $P(\mathbf{x}\mid y=k) = \mathcal{N}(\boldsymbol\mu_k, \Sigma_k)$.

**LDA** assumes a **shared** covariance $\Sigma_k = \Sigma$. The quadratic terms cancel between
classes, giving a **linear** decision boundary and the discriminant function:

$$\delta_k(\mathbf{x}) = \mathbf{x}^\top\Sigma^{-1}\boldsymbol\mu_k - \tfrac{1}{2}\boldsymbol\mu_k^\top\Sigma^{-1}\boldsymbol\mu_k + \log\pi_k$$

Predict $\arg\max_k \delta_k(\mathbf{x})$.

**QDA** allows per-class $\Sigma_k$ → **quadratic** boundaries, more parameters, needs more data.

LDA doubles as a **supervised dimensionality reduction** method: project onto at most $K-1$
dimensions that maximize the ratio of between-class to within-class scatter:

$$\max_\mathbf{w} \frac{\mathbf{w}^\top S_B\mathbf{w}}{\mathbf{w}^\top S_W\mathbf{w}}$$

(Unlike PCA, which is unsupervised and maximizes total variance.)

---

## 10. Generalized Linear Models

GLMs unify linear and logistic regression. Three components:

1. A **random component:** $y$ follows an exponential-family distribution.
2. A **systematic component:** the linear predictor $\eta = \mathbf{w}^\top\mathbf{x}$.
3. A **link function** $g$ connecting them: $g(\mathbb{E}[y]) = \eta$.

| Model | Distribution | Link | Inverse link |
|-------|-------------|------|--------------|
| Linear regression | Gaussian | Identity | $\mu = \eta$ |
| Logistic regression | Bernoulli | Logit | $\mu = \sigma(\eta)$ |
| **Poisson regression** | Poisson | Log | $\mu = e^\eta$ |
| **Gamma regression** | Gamma | Inverse/log | — |
| **Negative binomial** | NB | Log | — |

**Use Poisson regression for count targets** (number of purchases, claims, visits). Using
plain linear regression on counts predicts negative values and assumes constant variance,
whereas Poisson's variance equals its mean. If your counts are more dispersed than that
(variance > mean, called **overdispersion**), use negative binomial. Similarly, **Tweedie
regression** is the right tool for zero-inflated continuous targets (insurance claims,
purchase amounts).

---

## 11. Model comparison cheat sheet

| Algorithm | Type | Train | Predict | Scaling? | Interpretable | Best for |
|-----------|------|-------|---------|----------|---------------|----------|
| Linear regression | Parametric | $O(nd^2)$ | $O(d)$ | For regularization | ⭐⭐⭐⭐⭐ | Linear relationships, baselines |
| Logistic regression | Parametric | $O(nd\cdot\text{iters})$ | $O(d)$ | Yes | ⭐⭐⭐⭐⭐ | Binary classification baseline, calibrated probabilities |
| kNN | Non-parametric | $O(1)$ | $O(nd)$ | **Yes** | ⭐⭐⭐ | Small data, irregular boundaries |
| Naive Bayes | Parametric | $O(nd)$ | $O(Kd)$ | No | ⭐⭐⭐⭐ | Text, tiny data, speed |
| SVM (RBF) | Non-parametric | $O(n^2)$–$O(n^3)$ | $O(n_{SV}d)$ | **Yes** | ⭐⭐ | Medium data, clear margins, high-$d$ |
| Decision tree | Non-parametric | $O(nd\log n)$ | $O(\text{depth})$ | No | ⭐⭐⭐⭐⭐ | Explainability, rules |
| Random forest | Ensemble | $O(Bnd\log n)$ | $O(B\cdot\text{depth})$ | No | ⭐⭐⭐ | Strong general default, robust |
| Gradient boosting | Ensemble | $O(Mnd\log n)$ | $O(M\cdot\text{depth})$ | No | ⭐⭐⭐ | **Tabular SOTA** |
| Neural network | Parametric | Expensive | Fast | **Yes** | ⭐ | Images, text, audio, huge data |

### Decision guide

```
Tabular data?
├── < 1,000 rows      → Logistic/linear regression, Naive Bayes, small RF
├── 1k – 1M rows      → Gradient boosting (LightGBM/XGBoost/CatBoost)  ← start here
└── > 1M rows         → LightGBM, or linear models with SGD

Images/video          → CNN or Vision Transformer (fine-tune a pretrained one)
Text                  → Fine-tune a transformer; TF-IDF + linear as the baseline
Audio                 → Spectrogram + CNN, or a pretrained audio model
Time series           → Start with statistical (ARIMA/ETS) + gradient boosting on lag features
Sequential decisions  → Reinforcement learning
Need explanations     → Linear/tree, or any model + SHAP
Need probabilities    → Logistic regression, or calibrate whatever you use
```

---

## 12. Time series

Time series break the i.i.d. assumption, so they need their own toolkit.

### Core concepts

- **Stationarity:** the statistical properties (mean, variance, autocorrelation) don't change
  over time. Most classical methods require it. Test with **ADF** or **KPSS**; achieve it via
  differencing $y'_t = y_t - y_{t-1}$ and/or log transforms.
- **Autocorrelation (ACF):** $\rho_k = \text{Corr}(y_t, y_{t-k})$.
- **Partial autocorrelation (PACF):** correlation at lag $k$ with intermediate lags' effects
  removed.
- **Decomposition:** $y_t = \text{Trend}_t + \text{Seasonality}_t + \text{Residual}_t$
  (additive) or the multiplicative equivalent.

### Classical models

**AR($p$)** — regress on the past:

$$y_t = c + \sum_{i=1}^{p}\phi_i y_{t-i} + \epsilon_t$$

**MA($q$)** — regress on past errors:

$$y_t = \mu + \epsilon_t + \sum_{i=1}^{q}\theta_i\epsilon_{t-i}$$

**ARIMA($p,d,q$)** — AR + MA applied to the $d$-times-differenced series.
**SARIMA($p,d,q$)($P,D,Q$)$_s$** adds seasonal terms.

**Exponential smoothing / Holt-Winters:**

$$\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t$$

with extensions for trend (Holt) and seasonality (Holt-Winters).

### ML approach

Convert forecasting into supervised regression via **lag features**:

| Feature type | Examples |
|--------------|----------|
| Lags | $y_{t-1}, y_{t-2}, \dots, y_{t-k}$ |
| Rolling statistics | rolling mean/std/min/max over windows of 7, 30, 90 |
| Date parts | day-of-week, month, quarter, is_holiday |
| Cyclical encodings | $\sin/\cos$ of time-of-day, day-of-year |
| Exogenous | promotions, weather, price |

Then train LightGBM. **This wins most forecasting competitions.**

### Critical rules

1. **Never shuffle.** Use `TimeSeriesSplit` with expanding or rolling windows.
2. **Never use future information.** Rolling features must use only past windows
   (`.shift(1)` before `.rolling()`).
3. **Watch out for the horizon.** If you predict 7 days ahead, features must be available 7
   days ahead — lag them accordingly.
4. **Baseline = naive forecast** ($\hat y_{t+1} = y_t$) or seasonal naive
   ($\hat y_{t+1} = y_{t+1-s}$). A shocking number of complex models fail to beat it.
5. Deep learning options (N-BEATS, Temporal Fusion Transformer, PatchTST) help mainly with
   many related series; for a single series, classical + boosting usually wins.

---

## Self-check

1. Derive the normal equations from the least-squares objective.
2. Show that the softmax cross-entropy gradient w.r.t. logits is $\hat p - y$.
3. Explain why only support vectors matter, from complementary slackness.
4. What does the kernel trick buy you, and why does the dual make it possible?
5. Give the variance formula for a bagged ensemble and explain what random forests do to it.
6. Why does boosting overfit with more estimators while bagging doesn't?
7. When would you use Poisson regression instead of linear regression?
8. Name three ways a time-series pipeline leaks the future.

---

**Next:** [04 — Unsupervised Learning →](04_unsupervised_learning.md)
