# 04 — Unsupervised Learning

> Learning structure from data with no labels: clustering, dimensionality reduction, density
> estimation, and anomaly detection.

---

## Table of contents

1. [K-Means clustering](#1-k-means-clustering)
2. [Hierarchical clustering](#2-hierarchical-clustering)
3. [DBSCAN and density clustering](#3-dbscan-and-density-clustering)
4. [Gaussian Mixture Models & EM](#4-gaussian-mixture-models--em)
5. [Principal Component Analysis (PCA)](#5-principal-component-analysis-pca)
6. [Other matrix factorizations](#6-other-matrix-factorizations)
7. [Manifold learning: t-SNE and UMAP](#7-manifold-learning-t-sne-and-umap)
8. [Autoencoders](#8-autoencoders)
9. [Anomaly detection](#9-anomaly-detection)
10. [Association rules](#10-association-rule-learning)
11. [Evaluating unsupervised models](#11-evaluating-unsupervised-models)

---

## 1. K-Means Clustering

📄 `unsupervised/kmeans_clustering_example.py`

### Intuition
Partition $n$ points into $k$ groups so each point belongs to the cluster with the nearest
centroid.

### Objective

Minimize the **within-cluster sum of squares** (inertia):

$$J = \sum_{j=1}^{k}\sum_{\mathbf{x}\in C_j}\|\mathbf{x}-\boldsymbol\mu_j\|_2^2$$

where $\boldsymbol\mu_j = \frac{1}{|C_j|}\sum_{\mathbf{x}\in C_j}\mathbf{x}$ is the centroid.

This is NP-hard in general, so we use a greedy alternating algorithm.

### Lloyd's algorithm

```
1. Initialize k centroids (k-means++ recommended)
2. Repeat until assignments stop changing:
   a. ASSIGN:  each point → nearest centroid
               c_i = argmin_j ||x_i - μ_j||²
   b. UPDATE:  each centroid → mean of its assigned points
               μ_j = (1/|C_j|) Σ_{i: c_i = j} x_i
```

**Convergence guarantee:** both steps monotonically decrease $J$ (the assign step by
definition; the update step because the mean minimizes squared distance). $J$ is bounded below
and there are finitely many partitions, so the algorithm terminates. **But it converges to a
local optimum**, which depends on initialization — hence `n_init=10` (run several times, keep
the best $J$).

This is **coordinate descent** on $J$, alternating between assignments and centroids — exactly
the same structure as the EM algorithm in §4.

### k-means++ initialization

Random initialization can be terrible. k-means++ spreads the initial centroids:

```
1. Choose the first centroid uniformly at random from the data
2. For each remaining centroid:
   - compute D(x) = distance from x to its nearest chosen centroid
   - choose the next centroid from the data with probability ∝ D(x)²
```

This gives an $O(\log k)$ approximation guarantee in expectation. It's the default and you
should keep it.

### Choosing k

| Method | How |
|--------|-----|
| **Elbow method** | Plot $J$ vs $k$; look for the "elbow" where returns diminish. Subjective |
| **Silhouette score** | See below; pick $k$ maximizing the mean silhouette |
| **Gap statistic** | Compare $\log J$ to that expected under a null uniform reference |
| **Davies–Bouldin** | Ratio of within-cluster scatter to between-cluster separation; lower is better |
| **Domain knowledge** | Usually the best answer — you often *know* how many segments you want |

**Silhouette coefficient** for point $i$:

$$s(i) = \frac{b(i)-a(i)}{\max\{a(i), b(i)\}} \in [-1,1]$$

where $a(i)$ = mean distance to points in its own cluster (cohesion), $b(i)$ = mean distance
to points in the nearest *other* cluster (separation). Near 1 = well-clustered; near 0 = on a
boundary; negative = probably in the wrong cluster.

### Assumptions and limitations

| Assumption | What breaks when violated |
|------------|--------------------------|
| Clusters are spherical (isotropic) | Elongated/elliptical clusters get split |
| Clusters have similar size and density | Large clusters get carved up, small ones absorbed |
| Euclidean distance is meaningful | Fails in very high dimensions |
| You know $k$ | Wrong $k$ = meaningless clusters |
| No outliers | Means are pulled badly by outliers |

**Variants:** **K-medoids/PAM** (uses actual data points as centers, robust to outliers),
**MiniBatchKMeans** (scales to millions of rows), **Fuzzy c-means** (soft memberships),
**Spectral clustering** (clusters in the eigenspace of a similarity graph — handles non-convex
shapes), **Kernel k-means**.

**Always scale features before clustering.** Complexity: $O(nkdi)$ for $i$ iterations.

---

## 2. Hierarchical Clustering

Builds a **tree of clusters** (dendrogram) — no need to pre-specify $k$.

**Agglomerative (bottom-up):** start with every point as its own cluster; repeatedly merge the
two closest clusters until one remains. **Divisive (top-down)** does the reverse.

### Linkage criteria — how to measure distance between *clusters*

| Linkage | Formula | Behaviour |
|---------|---------|-----------|
| **Single** | $\min_{a\in A, b\in B} d(a,b)$ | Finds elongated/chained clusters; sensitive to noise |
| **Complete** | $\max_{a\in A,b\in B}d(a,b)$ | Compact, equal-diameter clusters |
| **Average** | $\frac{1}{\|A\|\|B\|}\sum_{a,b}d(a,b)$ | Compromise |
| **Ward** | Minimizes the increase in total within-cluster variance | **Best default**; produces balanced, spherical clusters |

**Ward's increase in SSE when merging $A$ and $B$:**

$$\Delta = \frac{|A||B|}{|A|+|B|}\|\boldsymbol\mu_A - \boldsymbol\mu_B\|^2$$

### Reading a dendrogram
The y-axis is the merge distance. Cut the tree horizontally at a chosen height to get clusters;
the biggest vertical gap between merges suggests a natural cut.

**Pros:** no $k$ needed upfront, gives a full hierarchy, deterministic.
**Cons:** $O(n^2)$ memory and $O(n^3)$ (or $O(n^2\log n)$) time — impractical beyond ~10k
points; merges are greedy and irreversible.

---

## 3. DBSCAN and Density Clustering

**Density-Based Spatial Clustering of Applications with Noise** — finds clusters as dense
regions separated by sparse ones.

### Definitions

Two hyperparameters: $\varepsilon$ (neighbourhood radius) and `min_samples` (density threshold).

- **Core point:** has ≥ `min_samples` points within distance $\varepsilon$.
- **Border point:** within $\varepsilon$ of a core point, but not itself core.
- **Noise point:** neither — labeled $-1$ and **not assigned to any cluster**.

### Algorithm

```
for each unvisited point p:
    mark visited
    N = neighbors within ε
    if |N| < min_samples: label p as noise (may be relabeled as border later)
    else:
        start a new cluster; add p
        expand: for each q in N, if q is core, add q's neighborhood to N
```

### Why it's valuable

| ✅ | ❌ |
|---|---|
| **Doesn't need $k$** | Very sensitive to $\varepsilon$ |
| Finds **arbitrarily shaped** clusters | Struggles when clusters have varying density |
| **Explicitly identifies outliers** | Suffers in high dimensions |
| Robust to noise | Border-point assignment depends on processing order |

**Choosing $\varepsilon$:** plot the sorted distance to each point's $k$-th nearest neighbour
(k = `min_samples`); the "knee" in that curve is a good $\varepsilon$.
**`min_samples` rule of thumb:** $\ge d+1$, commonly $2d$.

**HDBSCAN** is the modern successor: it varies $\varepsilon$ across the space, handling
variable-density clusters, and only needs `min_cluster_size`. **Prefer HDBSCAN in new work.**

**Mean Shift** is another density method: iteratively shift each point toward the local mode of
a kernel density estimate. Finds the number of clusters automatically but is $O(n^2)$.

---

## 4. Gaussian Mixture Models & EM

### The model

Assume the data comes from a mixture of $K$ Gaussians:

$$p(\mathbf{x}) = \sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k,\Sigma_k), \qquad \sum_k \pi_k = 1,\;\; \pi_k \ge 0$$

$\pi_k$ are the **mixing coefficients** (prior probability of each component). Unlike K-Means,
GMM does **soft assignment** — each point gets a probability of belonging to each cluster — and
can model elliptical, differently-oriented clusters via full covariance matrices.

### Expectation–Maximization (EM)

We can't maximize the log-likelihood directly (the sum inside the log has no closed form), so
we introduce latent assignment variables and alternate:

**E-step** — compute **responsibilities** (posterior probability that component $k$ generated
point $i$):

$$\gamma_{ik} = \frac{\pi_k\,\mathcal{N}(\mathbf{x}_i\mid\boldsymbol\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}_i\mid\boldsymbol\mu_j,\Sigma_j)}$$

**M-step** — update parameters as responsibility-weighted statistics, with $N_k = \sum_i\gamma_{ik}$:

$$\pi_k = \frac{N_k}{n}, \qquad \boldsymbol\mu_k = \frac{1}{N_k}\sum_{i=1}^n\gamma_{ik}\mathbf{x}_i, \qquad \Sigma_k = \frac{1}{N_k}\sum_{i=1}^n\gamma_{ik}(\mathbf{x}_i-\boldsymbol\mu_k)(\mathbf{x}_i-\boldsymbol\mu_k)^\top$$

Repeat until the log-likelihood converges.

**Guarantee:** EM monotonically increases the log-likelihood (it maximizes a lower bound, the
ELBO, at each step) — but converges to a **local** optimum. Initialize with K-Means.

> **K-Means is the limiting case of GMM** with $\Sigma_k = \sigma^2 I$ as $\sigma^2\to0$:
> responsibilities become hard 0/1 assignments. Understanding this connection makes both
> algorithms click.

### Covariance types

| Type | Parameters per component | Shape |
|------|--------------------------|-------|
| `spherical` | 1 | Circles of varying size |
| `diag` | $d$ | Axis-aligned ellipses |
| `tied` | shared $d\times d$ | All components share one shape |
| `full` | $d(d+1)/2$ | Arbitrary ellipses (most flexible, most data-hungry) |

### Choosing K
Use information criteria that penalize complexity:

$$\text{AIC} = 2p - 2\log L, \qquad \text{BIC} = p\log n - 2\log L$$

($p$ = number of parameters). Lower is better; BIC penalizes more strongly and tends to pick
simpler models. Also consider a **Dirichlet Process GMM** (`BayesianGaussianMixture`), which
infers the effective number of components automatically.

**GMM is also a density estimator**, so it gives you $p(\mathbf{x})$ — useful directly for
anomaly detection.

---

## 5. Principal Component Analysis (PCA)

📄 `unsupervised/pca_example.py`

### Intuition
Find the directions of maximum variance in the data and project onto them. It's a rotation to
a new coordinate system where the axes are ordered by how much information they carry.

### Derivation (maximum variance)

Assume the data is **centered** ($\boldsymbol\mu = 0$; this is mandatory). We want the unit
vector $\mathbf{w}$ maximizing the variance of the projections $\mathbf{w}^\top\mathbf{x}_i$:

$$\text{Var} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{w}^\top\mathbf{x}_i)^2 = \mathbf{w}^\top\left(\frac{1}{n}X^\top X\right)\mathbf{w} = \mathbf{w}^\top\Sigma\mathbf{w}$$

Maximize subject to $\|\mathbf{w}\|=1$ via a Lagrange multiplier:

$$\mathcal{L} = \mathbf{w}^\top\Sigma\mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w}-1)$$
$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}} = 2\Sigma\mathbf{w} - 2\lambda\mathbf{w} = 0 \quad\Longrightarrow\quad \boxed{\Sigma\mathbf{w} = \lambda\mathbf{w}}$$

**That's the eigenvector equation.** The optimal direction is the eigenvector of the covariance
matrix, and the variance captured is its eigenvalue $\lambda$ (since
$\mathbf{w}^\top\Sigma\mathbf{w} = \lambda$). So take eigenvectors in descending eigenvalue
order. ∎

### Equivalent derivations
PCA also falls out of (a) **minimizing reconstruction error** $\sum_i\|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2$
for a linear projection, and (b) the **SVD** of the centered data matrix. All three give the
same answer — a good sign you're looking at something fundamental.

### Algorithm (via SVD — the numerically stable route)

```
1. Center:  X ← X - mean(X)          # REQUIRED
   (Standardize instead if features have different units)
2. SVD:     X = U Σ Vᵀ
3. Components = columns of V (the principal directions)
   Explained variance of component i = σ_i² / (n-1)
4. Project: Z = X V_k                # keep the first k components  → shape (n, k)
5. Reconstruct: X̂ = Z V_kᵀ + mean
```

Using SVD on $X$ avoids ever forming $X^\top X$, which squares the condition number.

### Explained variance ratio

$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j}$$

Choose $k$ so the cumulative EVR reaches a target (e.g. 95%), or look for the elbow in a scree
plot.

### Key properties
- Principal components are **orthogonal** and **uncorrelated** by construction.
- PCA is a **linear** method — it cannot unfold curved manifolds.
- Components are **not interpretable** as original features (each is a mix of all of them).
- **Scale-sensitive:** standardize when features have different units, or the largest-variance
  unit dominates.
- Discards low-variance directions — which is a *risk*, since low variance ≠ low importance
  for prediction. PCA is unsupervised and knows nothing about your target.

### Uses
Dimensionality reduction, decorrelation, visualization (first 2–3 PCs), noise reduction,
speeding up downstream models, and fixing multicollinearity (→ principal component regression).

**Variants:** **Kernel PCA** (non-linear via the kernel trick), **Incremental PCA**
(out-of-core), **Randomized/Truncated SVD** (fast approximate, works on sparse matrices),
**Sparse PCA** (interpretable, sparse loadings).

---

## 6. Other Matrix Factorizations

| Method | Constraint | Use |
|--------|-----------|-----|
| **Truncated SVD / LSA** | None; works on sparse data without centering | Text (TF-IDF matrices) |
| **NMF** | All entries $\ge 0$: $X \approx WH$ | Topic modelling, parts-based decomposition — non-negativity makes components additive and interpretable |
| **ICA** | Components statistically independent (not just uncorrelated) | Blind source separation (the "cocktail party problem"), EEG |
| **Factor Analysis** | Models shared latent factors + per-feature noise | Psychometrics, latent constructs |
| **Dictionary learning** | Sparse codes over a learned overcomplete basis | Signal/image denoising |
| **Matrix factorization (ALS/SGD)** | $R \approx UV^\top$ on observed entries only | Collaborative-filtering recommenders |

**NMF objective:**

$$\min_{W\ge0, H\ge0}\|X - WH\|_F^2$$

Because nothing can cancel out, NMF learns *parts* (e.g. facial features) rather than the
holistic, sign-mixed components PCA produces.

**Recommender matrix factorization:** with users $u$ and items $i$,
$\hat r_{ui} = \boldsymbol\mu + b_u + b_i + \mathbf{p}_u^\top\mathbf{q}_i$, trained on the
observed ratings with L2 regularization. The biases matter more than people expect.

---

## 7. Manifold Learning: t-SNE and UMAP

Non-linear dimensionality reduction, aimed at **visualization** rather than preprocessing.

### t-SNE (t-distributed Stochastic Neighbour Embedding)

**Idea:** convert distances into probabilities in both the high- and low-dimensional spaces,
then minimize the divergence between them.

High-dimensional similarity (Gaussian kernel, symmetrized):

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i-\mathbf{x}_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|\mathbf{x}_i-\mathbf{x}_k\|^2/2\sigma_i^2)}$$

Low-dimensional similarity uses a **Student-t distribution with 1 degree of freedom** (a Cauchy
— heavy tails):

$$q_{ij} = \frac{(1+\|\mathbf{y}_i-\mathbf{y}_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|\mathbf{y}_k-\mathbf{y}_l\|^2)^{-1}}$$

Minimize $\;C = D_{\text{KL}}(P\|Q) = \sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}$ by gradient descent.

**Why the heavy tail?** It solves the "crowding problem": there simply isn't enough room in 2-D
to preserve all the moderate distances from high-D. The t-distribution lets moderately distant
points sit much further apart in the embedding, freeing up space for clusters to separate.

**Perplexity** ($\approx$ effective number of neighbours, typically 5–50) sets each $\sigma_i$.

> **How to read a t-SNE plot — the rules everyone violates:**
> - Cluster **sizes mean nothing** (t-SNE expands dense clusters, contracts sparse ones).
> - **Distances between clusters mean nothing.**
> - Random "clusters" appear in pure noise at low perplexity.
> - Different runs/seeds give different pictures.
> - **It is a visualization tool, not a preprocessing step.** Don't feed t-SNE output to a model.

### UMAP (Uniform Manifold Approximation and Projection)

Based on Riemannian geometry and fuzzy simplicial sets; in practice:

| vs t-SNE | UMAP |
|----------|------|
| Speed | Much faster; scales to millions |
| Global structure | Better preserved (inter-cluster distances mean *something*) |
| New data | Can `transform()` unseen points; t-SNE cannot |
| Hyperparameters | `n_neighbors` (local↔global), `min_dist` (cluster tightness) |

**UMAP is generally the better default** for both visualization and as a preprocessing step
(it's used in embedding pipelines and RAG data exploration).

---

## 8. Autoencoders

📄 `unsupervised/autoencoder_example.py`

### Architecture

A neural network trained to reconstruct its own input through a bottleneck:

$$\mathbf{z} = f_{\text{enc}}(\mathbf{x}), \qquad \hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z}), \qquad \mathcal{L} = \|\mathbf{x}-\hat{\mathbf{x}}\|^2$$

The bottleneck $\mathbf{z}\in\mathbb{R}^k$ with $k \ll d$ is the **latent code / embedding**.
Because it must reconstruct the input from $k$ numbers, it is forced to learn the data's
essential structure.

> **A linear autoencoder with squared loss learns the same subspace as PCA.** The value of
> autoencoders comes entirely from non-linear activations.

### Variants

| Type | Mechanism | Use |
|------|-----------|-----|
| **Undercomplete** | $k < d$ bottleneck | Dimensionality reduction |
| **Denoising (DAE)** | Corrupt the input, reconstruct the clean original | Robust features, denoising |
| **Sparse** | L1 penalty or KL penalty on activations | Interpretable, overcomplete codes |
| **Contractive** | Penalize the encoder Jacobian $\\|J_f(\mathbf{x})\\|_F^2$ | Invariance to small perturbations |
| **Variational (VAE)** | Probabilistic latent space | **Generation** |
| **Masked (MAE)** | Mask image patches / tokens and reconstruct | Self-supervised pretraining |

### Variational Autoencoder (VAE)

The encoder outputs a *distribution* $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol\mu_\phi(\mathbf{x}), \text{diag}(\boldsymbol\sigma^2_\phi(\mathbf{x})))$
rather than a point. Train by maximizing the **Evidence Lower BOund (ELBO)**:

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)}_{\text{regularizer, prior } \mathcal{N}(0,I)}$$

The KL term pulls the latent distribution toward a standard normal, making the space
**continuous and sampleable** — so you can generate new data by drawing
$\mathbf{z}\sim\mathcal{N}(0,I)$ and decoding.

**Reparameterization trick** — you can't backpropagate through sampling, so rewrite:

$$\mathbf{z} = \boldsymbol\mu + \boldsymbol\sigma\odot\boldsymbol\epsilon, \qquad \boldsymbol\epsilon\sim\mathcal{N}(0,I)$$

Now the randomness sits in $\boldsymbol\epsilon$ (a constant w.r.t. the parameters) and
gradients flow through $\boldsymbol\mu$ and $\boldsymbol\sigma$. This trick is one of the most
important ideas in modern generative modelling.

**Closed-form KL** against a standard normal (what you actually implement):

$$D_{\text{KL}} = -\frac{1}{2}\sum_{j=1}^{k}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**$\beta$-VAE** scales the KL term by $\beta>1$ to encourage *disentangled* latent factors.

---

## 9. Anomaly Detection

Finding points that don't conform to expected behaviour: fraud, intrusion, defects, faults.

| Method | How | Notes |
|--------|-----|-------|
| **Z-score / IQR** | Univariate statistical thresholds | Baseline; misses multivariate anomalies |
| **Mahalanobis distance** | $\sqrt{(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)}$ | Accounts for correlation; assumes Gaussian |
| **GMM / KDE** | Low $p(\mathbf{x})$ = anomaly | Flexible densities |
| **Isolation Forest** | Random splits isolate anomalies in fewer splits | **Best general default**; $O(n\log n)$, scales well |
| **One-Class SVM** | Learn a boundary enclosing the normal data | Sensitive to `nu` and `gamma` |
| **Local Outlier Factor (LOF)** | Compare a point's local density to its neighbours' | Finds *local* anomalies |
| **Autoencoder reconstruction error** | Train on normal data; high error = anomaly | High-dimensional data (images, logs) |
| **Supervised** | Just train a classifier | **Use this if you have labels** — it beats all of the above |

**Isolation Forest scoring:**

$$s(\mathbf{x}, n) = 2^{-\frac{E[h(\mathbf{x})]}{c(n)}}, \qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$

where $h(\mathbf{x})$ is the path length to isolate $\mathbf{x}$ and $c(n)$ normalizes by the
average path length of an unsuccessful BST search. $s\to1$ = anomaly, $s\ll0.5$ = normal.
Anomalies are isolated near the root because they're few and different.

**Practical notes:** set `contamination` from your actual expected anomaly rate; evaluate with
PR-AUC (never accuracy); and remember that "anomalous" and "interesting" aren't the same thing
— you'll surface plenty of boring data-entry errors.

---

## 10. Association Rule Learning

Finding "customers who bought X also bought Y" patterns (market basket analysis).

For a rule $X \Rightarrow Y$:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Support** | $P(X\cap Y)$ | How often the itemset appears |
| **Confidence** | $P(Y\mid X) = \frac{P(X\cap Y)}{P(X)}$ | How often the rule holds |
| **Lift** | $\frac{P(X\cap Y)}{P(X)P(Y)}$ | >1 = positively associated; 1 = independent |
| **Conviction** | $\frac{1-P(Y)}{1-P(Y\mid X)}$ | How often the rule would be wrong if X and Y were independent |

**Apriori algorithm**, built on the *anti-monotone* property: if an itemset is infrequent, all
its supersets are infrequent — so you can prune the search space aggressively. **FP-Growth** is
the faster modern alternative (no candidate generation).

> **Watch lift, not confidence.** A rule with 80% confidence is worthless if the consequent
> occurs 80% of the time anyway — lift would be 1.0.

---

## 11. Evaluating Unsupervised Models

The hard part: with no labels, there is no ground truth.

### Internal metrics (no labels needed)

| Metric | Range | Good value |
|--------|-------|-----------|
| **Silhouette** | $[-1,1]$ | Higher |
| **Calinski–Harabasz** (variance ratio) | $[0,\infty)$ | Higher |
| **Davies–Bouldin** | $[0,\infty)$ | **Lower** |
| **Inertia (WCSS)** | $[0,\infty)$ | Lower — but always decreases with $k$, so use the elbow |

### External metrics (when you *do* have labels, e.g. for benchmarking)

| Metric | Notes |
|--------|-------|
| **Adjusted Rand Index (ARI)** | Agreement between two partitions, chance-corrected. 0 = random, 1 = perfect |
| **Normalized Mutual Information (NMI)** | Mutual information scaled to $[0,1]$ |
| **Homogeneity / Completeness / V-measure** | Each cluster contains one class / each class in one cluster / their harmonic mean |
| **Fowlkes–Mallows** | Geometric mean of pairwise precision and recall |

### The honest evaluation

For dimensionality reduction: **reconstruction error**, and whether downstream task performance
improves. For clustering: **does a domain expert find the clusters meaningful and actionable?**
Profile each cluster's feature distributions and give it a name. A clustering nobody can
interpret or act on has no value regardless of its silhouette score.

---

## Self-check

1. Prove that Lloyd's algorithm decreases the objective at every step.
2. Derive PCA from the maximum-variance objective. Why is centering required?
3. Explain how K-Means is a limiting case of GMM.
4. Write the E and M steps of EM for a GMM.
5. Why does t-SNE use a Student-t distribution in the low-dimensional space?
6. Name three things a t-SNE plot does *not* tell you.
7. Write the ELBO and explain the reparameterization trick.
8. When would you use DBSCAN over K-Means, and vice versa?

---

**Next:** [05 — Deep Learning →](05_deep_learning.md)
