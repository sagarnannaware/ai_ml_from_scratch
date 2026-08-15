# 01 — Mathematical Foundations

> Every piece of mathematics you need for machine learning, defined from scratch, with the
> ML use of each concept stated explicitly. Nothing here is decoration — if it's in this
> file, it shows up in an algorithm later.

**Prerequisites:** high-school algebra. That's genuinely it.

---

## Table of contents

- [Part 1 — Linear Algebra](#part-1--linear-algebra)
- [Part 2 — Calculus and Optimization Foundations](#part-2--calculus-and-optimization-foundations)
- [Part 3 — Probability](#part-3--probability)
- [Part 4 — Statistics](#part-4--statistics)
- [Part 5 — Optimization](#part-5--optimization)
- [Part 6 — Information Theory](#part-6--information-theory)
- [Part 7 — Numerical Computing](#part-7--numerical-computing)

---

# Part 1 — Linear Algebra

**Why it matters:** data is a matrix, models are matrix multiplications, and training is
moving through a high-dimensional vector space. Linear algebra *is* the language of ML.

## 1.1 Scalars, vectors, matrices, tensors

| Object | Definition | Notation | Example shape |
|--------|-----------|----------|---------------|
| **Scalar** | A single number | $a \in \mathbb{R}$ | `()` |
| **Vector** | An ordered list of $n$ numbers; a point/direction in $n$-dimensional space | $\mathbf{x} \in \mathbb{R}^n$ | `(n,)` |
| **Matrix** | A rectangular grid of numbers, $m$ rows × $n$ columns | $A \in \mathbb{R}^{m\times n}$ | `(m, n)` |
| **Tensor** | An array with any number of dimensions (*rank*/*order*) | $\mathcal{T} \in \mathbb{R}^{d_1 \times \cdots \times d_k}$ | `(d1,...,dk)` |

A vector:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n\end{bmatrix} \in \mathbb{R}^n$$

**Convention used throughout this guide:** vectors are *column* vectors. The element in
row $i$, column $j$ of $A$ is $A_{ij}$.

**In ML:** one training example is a vector $\mathbf{x}_i \in \mathbb{R}^d$ ($d$ features).
The whole dataset is the **design matrix** $X \in \mathbb{R}^{n \times d}$ — $n$ examples
(rows) by $d$ features (columns). A batch of RGB images is a rank-4 tensor
`(batch, channels, height, width)`.

## 1.2 Vector operations

**Addition** (element-wise): $(\mathbf{x} + \mathbf{y})_i = x_i + y_i$

**Scalar multiplication:** $(c\mathbf{x})_i = c\,x_i$

**Dot product (inner product)** — the single most important operation in ML:

$$\mathbf{x}^\top \mathbf{y} = \langle \mathbf{x}, \mathbf{y}\rangle = \sum_{i=1}^{n} x_i y_i \in \mathbb{R}$$

Geometric form:

$$\mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta$$

where $\theta$ is the angle between them. Consequences:
- $\mathbf{x}^\top\mathbf{y} = 0 \iff$ the vectors are **orthogonal** (perpendicular).
- $\mathbf{x}^\top\mathbf{y} > 0 \iff$ they point in broadly the same direction.

**In ML:** a linear model's prediction *is* a dot product, $\hat y = \mathbf{w}^\top\mathbf{x} + b$.
Attention scores are dot products. Embedding similarity is a (normalized) dot product.

**Outer product** — produces a matrix from two vectors:

$$\mathbf{x}\mathbf{y}^\top \in \mathbb{R}^{m\times n}, \qquad (\mathbf{x}\mathbf{y}^\top)_{ij} = x_i y_j$$

## 1.3 Norms (measuring size)

A **norm** $\|\cdot\|$ maps a vector to a non-negative "length" satisfying: $\|\mathbf{x}\|\ge 0$
with equality iff $\mathbf{x} = 0$; $\|c\mathbf{x}\| = |c|\|\mathbf{x}\|$; and the triangle
inequality $\|\mathbf{x}+\mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$.

**General $L^p$ norm:**

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

| Norm | Formula | ML meaning |
|------|---------|-----------|
| $L^1$ (Manhattan) | $\|\mathbf{x}\|_1 = \sum_i \|x_i\|$ | **Lasso** regularization → sparsity |
| $L^2$ (Euclidean) | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | **Ridge**/weight decay, distances, gradient clipping |
| $L^\infty$ (max) | $\|\mathbf{x}\|_\infty = \max_i \|x_i\|$ | Adversarial robustness budgets |
| $L^0$ ("norm") | count of non-zeros | Sparsity target (not a true norm; non-convex) |

**Frobenius norm** (for matrices): $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$ — the $L^2$ norm of
the flattened matrix. Used for weight decay on weight matrices.

**Squared $L^2$** $\|\mathbf{x}\|_2^2 = \mathbf{x}^\top\mathbf{x}$ is preferred in objectives
because it's differentiable everywhere (the square root's derivative blows up at 0) and its
gradient is simply $2\mathbf{x}$.

**Cosine similarity** — direction only, magnitude ignored:

$$\text{cos\_sim}(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|_2\|\mathbf{y}\|_2} \in [-1, 1]$$

This is the standard similarity measure for embeddings (RAG retrieval, semantic search).

## 1.4 Matrix multiplication

For $A \in \mathbb{R}^{m\times n}$ and $B \in \mathbb{R}^{n\times p}$, the product
$C = AB \in \mathbb{R}^{m\times p}$ has entries:

$$C_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$$

**Read it as:** $C_{ij}$ is the dot product of row $i$ of $A$ with column $j$ of $B$.
Inner dimensions must match; the outer dimensions survive: $(m\times \underline{n})(\underline{n}\times p) \to (m \times p)$.

**Properties:**
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B+C) = AB + AC$
- **NOT commutative**: $AB \ne BA$ in general
- Transpose reverses: $(AB)^\top = B^\top A^\top$
- Cost: $O(mnp)$ — this is why deep learning needs GPUs

**Transpose:** $(A^\top)_{ij} = A_{ji}$ — flip across the diagonal.

**In ML:** a full forward pass of a linear layer over a batch is one matmul:
$H = XW + \mathbf{b}$, where $X \in \mathbb{R}^{n\times d}$, $W \in \mathbb{R}^{d\times h}$.
99% of the FLOPs in a transformer are matrix multiplications.

## 1.5 Special matrices

| Matrix | Definition | Why it matters |
|--------|-----------|----------------|
| **Identity** $I$ | $I_{ii}=1$, else 0; $AI = A$ | Neutral element; residual connections |
| **Diagonal** $D$ | Non-zero only on diagonal | Cheap to invert: $D^{-1}_{ii} = 1/D_{ii}$; scaling |
| **Symmetric** | $A = A^\top$ | Covariance, kernel/Gram matrices; real eigenvalues |
| **Orthogonal** $Q$ | $Q^\top Q = QQ^\top = I$, so $Q^{-1}=Q^\top$ | Preserves lengths & angles; stable; used in init |
| **Positive definite** | $\mathbf{x}^\top A\mathbf{x} > 0\ \forall \mathbf{x}\ne 0$ | Guarantees a unique minimum; valid covariance/kernel |
| **Positive semi-definite (PSD)** | $\mathbf{x}^\top A\mathbf{x} \ge 0$ | Covariance matrices are always PSD |
| **Sparse** | Mostly zeros | Memory/compute savings; text features |

## 1.6 Linear independence, span, rank

- **Linear combination:** $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$.
- **Span:** the set of *all* linear combinations of a set of vectors — the subspace they generate.
- **Linearly independent:** no vector in the set can be written as a combination of the others.
  Formally, $\sum_i c_i\mathbf{v}_i = \mathbf{0}$ only when all $c_i = 0$.
- **Basis:** a linearly independent set that spans the space. $\mathbb{R}^n$ needs exactly $n$ basis vectors.
- **Rank** $\text{rank}(A)$: the number of linearly independent columns (= number of independent rows).
  - **Full rank:** $\text{rank}(A) = \min(m, n)$.
  - **Rank deficient:** there is redundancy — some feature is a combination of others.

**In ML:** if two features are perfectly correlated, $X$ is rank deficient, $X^\top X$ is
singular (non-invertible), and the normal-equation solution for linear regression *does not
exist uniquely*. This is **multicollinearity**. Ridge regression fixes it by adding
$\lambda I$, which forces full rank.

## 1.7 Matrix inverse and solving linear systems

$A^{-1}$ is the matrix with $AA^{-1} = A^{-1}A = I$. It exists iff $A$ is **square and full
rank** (equivalently: $\det(A) \ne 0$; equivalently: no zero eigenvalue).

Solving $A\mathbf{x} = \mathbf{b}$ gives $\mathbf{x} = A^{-1}\mathbf{b}$ — but **never
actually compute an inverse in code**. Use a solver:

```python
x = np.linalg.solve(A, b)      # ✅ faster, numerically stabler
x = np.linalg.inv(A) @ b       # ❌ avoid
```

**Determinant** $\det(A)$: the signed volume scaling factor of the linear map. $\det(A)=0$
means the transformation collapses space into a lower dimension (information is destroyed,
hence no inverse).

**Pseudo-inverse (Moore–Penrose)** $A^+$ — generalizes the inverse to non-square/singular
matrices. When $A$ has full column rank: $A^{+} = (A^\top A)^{-1}A^\top$. This is exactly
the least-squares solution operator.

**Trace:** $\text{tr}(A) = \sum_i A_{ii}$. Useful identity: $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$ (cyclic).

## 1.8 Eigenvalues and eigenvectors

An **eigenvector** of a square matrix $A$ is a non-zero vector whose *direction* is
unchanged by $A$; it is only scaled, by the **eigenvalue** $\lambda$:

$$A\mathbf{v} = \lambda\mathbf{v}, \qquad \mathbf{v}\ne\mathbf{0}$$

Found by solving the **characteristic equation**:

$$\det(A - \lambda I) = 0$$

**Eigendecomposition** (for a diagonalizable $A$):

$$A = V\Lambda V^{-1}$$

where $V$'s columns are the eigenvectors and $\Lambda = \text{diag}(\lambda_1,\dots,\lambda_n)$.
For a **symmetric** $A$ this becomes the **spectral decomposition**, with orthonormal eigenvectors:

$$A = Q\Lambda Q^\top, \qquad Q^\top Q = I$$

Useful facts: $\text{tr}(A) = \sum_i\lambda_i$, $\det(A) = \prod_i \lambda_i$, and $A$ is
positive definite iff all $\lambda_i > 0$.

**In ML:**
- **PCA** = eigendecomposition of the covariance matrix. Eigenvectors are the principal
  components; eigenvalues are the variance captured along each.
- The **Hessian**'s eigenvalues describe loss-surface curvature: all positive → local
  minimum; mixed signs → saddle point (the dominant obstacle in deep learning).
- The **condition number** $\kappa = |\lambda_{\max}| / |\lambda_{\min}|$ predicts how
  slowly gradient descent converges. Large $\kappa$ = elongated valley = zig-zagging.
- **Spectral radius** $\rho(A) = \max_i|\lambda_i|$ governs whether repeated multiplication
  (as in an RNN) explodes ($\rho>1$) or vanishes ($\rho<1$).

## 1.9 Singular Value Decomposition (SVD)

The most powerful factorization in applied linear algebra. **Every** matrix
$A \in \mathbb{R}^{m\times n}$ — square or not, singular or not — decomposes as:

$$A = U\Sigma V^\top$$

| Factor | Shape | Meaning |
|--------|-------|---------|
| $U$ | $m\times m$, orthogonal | Left singular vectors (basis of output space) |
| $\Sigma$ | $m\times n$, diagonal, $\sigma_1\ge\sigma_2\ge\cdots\ge 0$ | Singular values (stretch factors) |
| $V$ | $n\times n$, orthogonal | Right singular vectors (basis of input space) |

Interpretation: *any* linear map is a **rotation → scaling → rotation**.

Relationship to eigen: $\sigma_i = \sqrt{\lambda_i(A^\top A)}$, and $V$ holds the
eigenvectors of $A^\top A$, $U$ those of $AA^\top$.

**Truncated SVD / Eckart–Young theorem:** keeping the top $k$ singular values gives the
*provably best* rank-$k$ approximation in both Frobenius and spectral norm:

$$A_k = \sum_{i=1}^{k}\sigma_i \mathbf{u}_i\mathbf{v}_i^\top, \qquad \|A - A_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$$

**In ML:** PCA (numerically stable route), latent semantic analysis, recommender systems
(matrix factorization), image compression, and — directly — **LoRA**, which learns a
low-rank update $\Delta W = BA$ to a frozen weight matrix.

## 1.10 Matrix calculus (the gradient rules you'll actually use)

Let $\mathbf{x}\in\mathbb{R}^n$, $A$ constant. **Denominator layout** (gradient has the
shape of the variable):

| Function | Gradient |
|----------|----------|
| $f = \mathbf{a}^\top\mathbf{x}$ | $\nabla_\mathbf{x} f = \mathbf{a}$ |
| $f = \mathbf{x}^\top A\mathbf{x}$ | $\nabla_\mathbf{x} f = (A + A^\top)\mathbf{x}$; $=2A\mathbf{x}$ if $A$ symmetric |
| $f = \\|\mathbf{x}\\|_2^2 = \mathbf{x}^\top\mathbf{x}$ | $\nabla_\mathbf{x} f = 2\mathbf{x}$ |
| $f = \\|A\mathbf{x} - \mathbf{b}\\|_2^2$ | $\nabla_\mathbf{x} f = 2A^\top(A\mathbf{x}-\mathbf{b})$ |
| $\mathbf{y} = A\mathbf{x}$ | $\partial \mathbf{y}/\partial\mathbf{x} = A$ |
| $L$ scalar, $Y = XW$ | $\nabla_W L = X^\top \nabla_Y L$, $\quad\nabla_X L = (\nabla_Y L) W^\top$ |

That **last row is the backpropagation rule for a linear layer.** Memorize it.

---

# Part 2 — Calculus and Optimization Foundations

**Why it matters:** training = following gradients downhill. Backprop is the chain rule.

## 2.1 Derivatives

The **derivative** measures instantaneous rate of change:

$$f'(x) = \frac{df}{dx} = \lim_{h\to 0}\frac{f(x+h) - f(x)}{h}$$

Geometrically: the slope of the tangent line.

**Rules you need:**

| Rule | Formula |
|------|---------|
| Power | $\frac{d}{dx}x^n = nx^{n-1}$ |
| Exponential | $\frac{d}{dx}e^x = e^x$ |
| Logarithm | $\frac{d}{dx}\ln x = \frac{1}{x}$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ |
| **Chain** | $\frac{d}{dx}f(g(x)) = f'(g(x))\cdot g'(x)$ |

**The chain rule is backpropagation.** Everything else is bookkeeping.

## 2.2 Partial derivatives and the gradient

For $f(x_1,\dots,x_n)$, the **partial derivative** $\frac{\partial f}{\partial x_i}$ is the
derivative with respect to $x_i$ holding all others fixed.

The **gradient** collects them into a vector:

$$\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n}\end{bmatrix} \in \mathbb{R}^n$$

**Key property:** $\nabla f$ points in the direction of **steepest ascent**, and
$\|\nabla f\|$ is the rate of increase in that direction.

*Why?* The directional derivative along unit vector $\mathbf{u}$ is
$D_\mathbf{u}f = \nabla f^\top \mathbf{u} = \|\nabla f\|\cos\theta$, maximized when
$\theta = 0$, i.e. $\mathbf{u}$ parallel to $\nabla f$.

**Therefore gradient *descent* steps in $-\nabla f$.** That's the entire justification for
the update rule $\theta \leftarrow \theta - \eta\nabla_\theta L$.

## 2.3 Jacobian and Hessian

**Jacobian** — for a vector-valued $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$, the matrix of
all first partials:

$$J = \frac{\partial \mathbf{f}}{\partial\mathbf{x}} \in \mathbb{R}^{m\times n}, \qquad J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Hessian** — the matrix of second partials of a scalar function:

$$H = \nabla^2 f \in \mathbb{R}^{n\times n}, \qquad H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

$H$ is symmetric (for nice $f$) and describes **curvature**:

| Hessian eigenvalues | Point type |
|---------------------|-----------|
| All $> 0$ (positive definite) | Local **minimum** |
| All $< 0$ | Local **maximum** |
| Mixed signs | **Saddle point** |
| Some $= 0$ | Inconclusive (plateau) |

**In ML:** in high dimensions, saddle points vastly outnumber local minima — which is why
deep learning works despite non-convexity. Second-order methods (Newton) use $H$ but
$O(n^2)$ storage is impossible for billions of parameters; hence first-order methods (Adam)
that cheaply approximate curvature.

## 2.4 Taylor series

Approximating a function near a point $a$:

$$f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

Multivariate second-order form (the backbone of optimization theory):

$$f(\mathbf{x} + \Delta) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\Delta + \tfrac{1}{2}\Delta^\top H \Delta$$

Minimizing the right-hand side over $\Delta$ gives **Newton's method**:
$\Delta = -H^{-1}\nabla f$.

## 2.5 Convexity

A set is **convex** if the line segment between any two of its points stays inside it.
A function is **convex** if:

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \le \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y}), \quad \forall \lambda\in[0,1]$$

i.e. the chord lies above the curve. Equivalent tests: $f'' \ge 0$ (1-D), or $H \succeq 0$
(Hessian PSD, multivariate).

**Why you care:** for a convex function, **any local minimum is the global minimum**, and
gradient descent is guaranteed to find it.

| Convex | Non-convex |
|--------|-----------|
| Linear regression (MSE), ridge, lasso, logistic regression, SVM | Neural networks, K-Means, GMM, matrix factorization |

Deep learning is non-convex — we accept "good enough" minima and rely on the empirical fact
that most minima found by SGD generalize well.

## 2.6 Lagrange multipliers (constrained optimization)

To minimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$, form the **Lagrangian**:

$$\mathcal{L}(\mathbf{x},\lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

and solve $\nabla_\mathbf{x}\mathcal{L}= 0$, $\nabla_\lambda\mathcal{L} = 0$. Intuition: at
the optimum, the gradients of $f$ and $g$ are parallel.

With inequality constraints $g_i(\mathbf{x}) \le 0$, optimality requires the **KKT
conditions**: stationarity, primal feasibility, dual feasibility ($\alpha_i \ge 0$), and
**complementary slackness** $\alpha_i g_i(\mathbf{x}) = 0$.

**In ML:** the SVM dual is derived exactly this way; complementary slackness is *why* only
support vectors have non-zero $\alpha_i$. Ridge regression is the Lagrangian form of
"minimize MSE subject to $\|\mathbf{w}\|_2^2 \le t$."

---

# Part 3 — Probability

**Why it matters:** data is noisy, models are uncertain, and losses are negative
log-likelihoods. Probability is how ML handles not-knowing.

## 3.1 Basic definitions

- **Sample space** $\Omega$: all possible outcomes.
- **Event** $A \subseteq \Omega$: a set of outcomes.
- **Probability** $P(A) \in [0,1]$, with $P(\Omega) = 1$.
- **Random variable (RV)** $X$: a function mapping outcomes to numbers.
  - *Discrete*: countable values (dice, class labels).
  - *Continuous*: uncountable values (height, pixel intensity).

**Axioms (Kolmogorov):** (1) $P(A)\ge0$; (2) $P(\Omega)=1$; (3) for disjoint events,
$P(\bigcup_i A_i) = \sum_i P(A_i)$.

## 3.2 Distributions

**PMF** (discrete): $p(x) = P(X = x)$, with $\sum_x p(x) = 1$.

**PDF** (continuous): $p(x) \ge 0$ with $\int_{-\infty}^{\infty}p(x)dx = 1$. Note
$P(X = x) = 0$ for continuous RVs; only intervals have probability:
$P(a\le X\le b) = \int_a^b p(x)dx$. A PDF *can* exceed 1.

**CDF:** $F(x) = P(X \le x)$, non-decreasing from 0 to 1.

## 3.3 Joint, marginal, conditional

| Concept | Definition |
|---------|-----------|
| **Joint** | $P(X, Y)$ — both happen |
| **Marginal** | $P(X) = \sum_y P(X, Y=y)$ (or $\int P(x,y)dy$) — "sum out" the other variable |
| **Conditional** | $P(X\mid Y) = \dfrac{P(X,Y)}{P(Y)}$ — probability of $X$ *given* $Y$ is known |

**Product / chain rule:**

$$P(X_1,\dots,X_n) = P(X_1)P(X_2\mid X_1)P(X_3\mid X_1,X_2)\cdots P(X_n\mid X_1,\dots,X_{n-1})$$

**This is exactly what an autoregressive language model computes** — the probability of a
sentence factored into next-token predictions.

**Independence:** $X \perp Y \iff P(X,Y) = P(X)P(Y) \iff P(X\mid Y) = P(X)$.

**Conditional independence:** $X \perp Y \mid Z \iff P(X,Y\mid Z) = P(X\mid Z)P(Y\mid Z)$.
This is the "naive" assumption in Naive Bayes.

## 3.4 Bayes' theorem

$$\boxed{P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)}}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(H\mid E)$ | **Posterior** | Belief in hypothesis after seeing evidence |
| $P(E\mid H)$ | **Likelihood** | How well the hypothesis explains the evidence |
| $P(H)$ | **Prior** | Belief before seeing evidence |
| $P(E)$ | **Evidence / marginal likelihood** | $\sum_H P(E\mid H)P(H)$ — normalizer |

**Worked example (the classic that everyone gets wrong):** A disease affects 1% of people.
A test is 99% sensitive and 99% specific. You test positive. What's the probability you're sick?

$$P(D\mid +) = \frac{0.99 \times 0.01}{0.99\times0.01 + 0.01\times0.99} = \frac{0.0099}{0.0198} = 0.5$$

**50%, not 99%.** Base rates dominate. This is the same reason accuracy is a terrible metric
for imbalanced classification.

## 3.5 Expectation, variance, covariance

**Expectation** (mean) — the probability-weighted average:

$$\mathbb{E}[X] = \sum_x x\,p(x) \qquad \text{or} \qquad \mathbb{E}[X] = \int x\,p(x)\,dx$$

*Linearity* (always true, even for dependent variables):
$\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$

**Law of the unconscious statistician:** $\mathbb{E}[g(X)] = \sum_x g(x)p(x)$.

**Variance** — expected squared deviation from the mean:

$$\text{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

$\text{Var}(aX + b) = a^2\text{Var}(X)$. **Standard deviation:** $\sigma = \sqrt{\text{Var}(X)}$.

**Covariance** — how two variables move together:

$$\text{Cov}(X,Y) = \mathbb{E}\big[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\big] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

**Correlation** — covariance normalized to $[-1,1]$:

$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}$$

> Independence $\Rightarrow$ zero covariance. **The converse is false**: $Y = X^2$ with
> $X\sim\mathcal{N}(0,1)$ has $\text{Cov}=0$ but total dependence. Correlation only detects
> *linear* relationships.

**Covariance matrix** for $\mathbf{x}\in\mathbb{R}^d$:

$$\Sigma = \mathbb{E}\big[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^\top\big] \in \mathbb{R}^{d\times d}$$

Symmetric, PSD. Diagonal entries are variances; off-diagonals are covariances. **PCA
eigendecomposes exactly this matrix.**

## 3.6 Key distributions

| Distribution | PMF/PDF | Mean | Variance | ML use |
|---|---|---|---|---|
| **Bernoulli**($p$) | $p^x(1-p)^{1-x}$, $x\in\\{0,1\\}$ | $p$ | $p(1-p)$ | Binary labels; logistic regression's likelihood |
| **Binomial**($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Counts of successes; A/B tests |
| **Categorical**($\mathbf{p}$) | $\prod_k p_k^{[x=k]}$ | — | — | Multiclass labels; softmax output |
| **Poisson**($\lambda$) | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Event counts; rare events |
| **Uniform**($a,b$) | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Random init, sampling |
| **Gaussian**($\mu,\sigma^2$) | see below | $\mu$ | $\sigma^2$ | Noise model, init, VAEs, diffusion |
| **Exponential**($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Waiting times, survival |
| **Beta**($\alpha,\beta$) | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | — | Conjugate prior for $p$; Thompson sampling |

**Gaussian (Normal) — the one you must know cold:**

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Multivariate version, $\mathbf{x}\in\mathbb{R}^d$:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\right)$$

The exponent's quadratic form $(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$
is the squared **Mahalanobis distance** — a distance that accounts for correlation and scale.
It's the principled way to do anomaly detection on continuous features.

**Why Gaussians are everywhere:** the CLT makes them the natural noise model; they're the
maximum-entropy distribution for a fixed mean and variance (least assumptions); and they're
closed under linear maps, marginalization, and conditioning, which makes the algebra tractable.

## 3.7 Central Limit Theorem & Law of Large Numbers

**LLN:** the sample mean converges to the true mean as $n\to\infty$:
$\bar{X}_n \to \mu$.

**CLT:** for i.i.d. $X_i$ with mean $\mu$ and finite variance $\sigma^2$, the *standardized*
sample mean converges to a standard normal regardless of the original distribution:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$$

Equivalently $\bar X_n \approx \mathcal{N}(\mu, \sigma^2/n)$. The
$\sigma/\sqrt{n}$ is the **standard error** — note the $\sqrt{n}$: to halve your uncertainty
you need **4×** the data. This governs how big your validation set must be.

## 3.8 Maximum Likelihood Estimation (MLE)

Given data $\mathcal{D}=\{x_1,\dots,x_n\}$ assumed i.i.d. from $p(x\mid\theta)$, the
**likelihood** is:

$$L(\theta) = \prod_{i=1}^{n}p(x_i\mid\theta)$$

We maximize the **log-likelihood** instead (monotonic, turns products into sums, avoids
numerical underflow):

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^{n}\log p(x_i\mid\theta) = \arg\min_\theta \underbrace{-\sum_{i=1}^{n}\log p(x_i\mid\theta)}_{\text{negative log-likelihood (NLL)}}$$

> **The single most important conceptual bridge in ML:** *loss functions are negative log
> likelihoods.*
> - Gaussian noise → MLE **is** least squares (MSE).
> - Bernoulli labels → MLE **is** binary cross-entropy.
> - Categorical labels → MLE **is** softmax cross-entropy.
>
> You are not choosing losses arbitrarily; you are choosing a noise model.

**Derivation — MSE from Gaussian noise.** Assume $y_i = \mathbf{w}^\top\mathbf{x}_i + \epsilon_i$,
$\epsilon_i \sim \mathcal{N}(0,\sigma^2)$. Then:

$$\log L = \sum_i \log\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - \mathbf{w}^\top\mathbf{x}_i)^2}{2\sigma^2}\right) = C - \frac{1}{2\sigma^2}\sum_i (y_i - \mathbf{w}^\top\mathbf{x}_i)^2$$

Maximizing over $\mathbf{w}$ = minimizing $\sum_i(y_i - \mathbf{w}^\top\mathbf{x}_i)^2$. **That is
MSE.** ∎

## 3.9 MAP and Bayesian inference

**Maximum a posteriori** adds a prior:

$$\hat\theta_{\text{MAP}} = \arg\max_\theta \; p(\mathcal{D}\mid\theta)p(\theta) = \arg\max_\theta\left[\log p(\mathcal{D}\mid\theta) + \log p(\theta)\right]$$

> **Second important bridge:** *regularization is a prior.*
> - Gaussian prior $\theta\sim\mathcal{N}(0,\tau^2)$ → $L^2$ penalty → **ridge regression**.
> - Laplace prior $\theta\sim\text{Laplace}(0,b)$ → $L^1$ penalty → **lasso**.

Full **Bayesian inference** keeps the whole posterior instead of a point estimate, and
predicts by integrating over it:

$$p(y\mid x,\mathcal{D}) = \int p(y\mid x,\theta)\,p(\theta\mid\mathcal{D})\,d\theta$$

Expensive but gives calibrated uncertainty. Practical approximations: MCMC, variational
inference, Laplace approximation, deep ensembles, MC dropout.

---

# Part 4 — Statistics

## 4.1 Descriptive statistics

| Statistic | Formula | Notes |
|-----------|---------|-------|
| Sample mean | $\bar{x} = \frac{1}{n}\sum_i x_i$ | Sensitive to outliers |
| Median | middle value | Robust |
| Sample variance | $s^2 = \frac{1}{n-1}\sum_i(x_i-\bar x)^2$ | $n-1$ = **Bessel's correction** (unbiasedness) |
| Skewness | $\mathbb{E}[(\frac{X-\mu}{\sigma})^3]$ | Asymmetry |
| Kurtosis | $\mathbb{E}[(\frac{X-\mu}{\sigma})^4]$ | Tail heaviness |
| IQR | $Q_3 - Q_1$ | Outliers: outside $Q_1 - 1.5\,\text{IQR}$, $Q_3 + 1.5\,\text{IQR}$ |

**Why $n-1$?** Using $\bar x$ (estimated from the same data) instead of the true $\mu$
systematically underestimates spread. Dividing by $n-1$ corrects the bias exactly: you spent
one degree of freedom estimating the mean.

## 4.2 Estimators, bias, variance

An **estimator** $\hat\theta$ is a function of data used to guess a parameter.

- **Bias:** $\text{Bias}(\hat\theta) = \mathbb{E}[\hat\theta] - \theta$
- **Variance:** $\text{Var}(\hat\theta) = \mathbb{E}[(\hat\theta - \mathbb{E}[\hat\theta])^2]$
- **Mean squared error:**

$$\text{MSE}(\hat\theta) = \mathbb{E}[(\hat\theta - \theta)^2] = \text{Bias}(\hat\theta)^2 + \text{Var}(\hat\theta)$$

This decomposition is the ancestor of the **bias–variance tradeoff** in [Module 02](02_ml_fundamentals.md#3-the-biasvariance-tradeoff).
A biased estimator with lower variance can beat an unbiased one — that's exactly what ridge
regression does.

## 4.3 Confidence intervals

A 95% CI is an interval computed from data such that, over repeated sampling, 95% of such
intervals contain the true parameter. For a mean with large $n$:

$$\bar{x} \pm z_{\alpha/2}\frac{s}{\sqrt{n}}, \qquad z_{0.025} = 1.96$$

> **It does NOT mean** "95% probability the parameter is in *this* interval" — the parameter
> is fixed, the interval is random. (The Bayesian analogue, the *credible interval*, does
> have that interpretation.)

## 4.4 Hypothesis testing

1. **Null hypothesis** $H_0$: the boring default (no effect, model B ≡ model A).
2. **Alternative** $H_1$.
3. **Test statistic** — a number summarizing the evidence.
4. **p-value** — $P(\text{observing data at least this extreme} \mid H_0 \text{ true})$.
5. Reject $H_0$ if $p < \alpha$ (typically 0.05).

| | $H_0$ true | $H_0$ false |
|---|---|---|
| **Reject $H_0$** | Type I error ($\alpha$), false positive | ✅ Correct (power = $1-\beta$) |
| **Fail to reject** | ✅ Correct | Type II error ($\beta$), false negative |

**A p-value is not** the probability that $H_0$ is true, nor the probability your result was
chance, nor a measure of effect size.

**Multiple comparisons:** testing $m$ hypotheses at $\alpha$ inflates the family-wise error
rate to $1-(1-\alpha)^m$. Correct with **Bonferroni** ($\alpha/m$, conservative) or
**Benjamini–Hochberg** (controls false discovery rate; better for large $m$).

**In ML:** comparing two models on the same test set is a paired test (McNemar's for
classifiers). If you evaluate 100 model variants on one validation set, you have a multiple
comparisons problem — the winner is partly winning by luck. This is why you need a *held-out
test set touched once*.

## 4.5 Common tests

| Test | Use when |
|------|----------|
| **t-test** (one/two-sample, paired) | Comparing means, small $n$, roughly normal |
| **z-test** | Comparing means/proportions, large $n$ |
| **Chi-squared** | Independence of categorical variables; goodness of fit |
| **ANOVA** | Comparing means across 3+ groups |
| **Mann–Whitney U** | Non-parametric alternative to t-test |
| **Kolmogorov–Smirnov** | Do two samples come from the same distribution? (**drift detection**) |
| **McNemar's** | Comparing two classifiers on the same test set |

## 4.6 Bootstrap

Resample the data *with replacement* $B$ times, recompute the statistic each time, and use
the spread of those $B$ values as your uncertainty estimate. Requires no distributional
assumptions.

```
for b in 1..B:
    D_b = sample n points from D with replacement
    θ_b = statistic(D_b)
CI = [percentile(θ, 2.5), percentile(θ, 97.5)]
```

**In ML:** confidence intervals on any metric (AUC, F1) where no closed form exists. Also the
statistical basis of **bagging** — each bootstrap sample omits about $1/e \approx 36.8\%$ of
the data, which becomes the "out-of-bag" validation set in random forests.

## 4.7 Correlation vs causation

Correlation between $X$ and $Y$ admits four explanations: $X\to Y$; $Y\to X$; a confounder
$Z\to X, Z\to Y$; or coincidence/selection bias.

**Simpson's paradox:** a trend present in every subgroup can reverse when groups are pooled.

**Establishing causation** needs randomization (A/B tests) or causal-inference machinery
(instrumental variables, difference-in-differences, regression discontinuity, propensity
score matching, do-calculus).

**In ML:** models learn correlations. A hospital model that learned "asthma → lower pneumonia
risk" was picking up that asthmatics get aggressive early treatment. Deployed as a triage
tool, it would kill people. Always ask what a feature *means*.

---

# Part 5 — Optimization

## 5.1 The problem

$$\theta^* = \arg\min_{\theta\in\mathbb{R}^d} J(\theta)$$

where $J$ is the objective/loss/cost. A **stationary point** has $\nabla J(\theta) = 0$.

## 5.2 Gradient descent

$$\theta_{t+1} = \theta_t - \eta\,\nabla_\theta J(\theta_t)$$

$\eta$ is the **learning rate / step size** — the most important hyperparameter in ML.

| $\eta$ too small | $\eta$ too large |
|---|---|
| Slow convergence, may stall | Overshoot, oscillate, or diverge to NaN |

For a convex $L$-smooth function, $\eta \le 1/L$ guarantees convergence, where $L$ is the
largest Hessian eigenvalue.

## 5.3 Batch, stochastic, and mini-batch

| Variant | Gradient computed on | Update cost | Noise | Notes |
|---------|---------------------|-------------|-------|-------|
| **Batch GD** | All $n$ examples | $O(n)$/step | None | Exact but slow; infeasible at scale |
| **SGD** | 1 example | $O(1)$ | High | Fast, noisy; noise can escape saddle points |
| **Mini-batch SGD** | $B$ examples (32–512 typical) | $O(B)$ | Medium | **The standard.** GPU-efficient |

$$\theta_{t+1} = \theta_t - \eta\cdot\frac{1}{B}\sum_{i\in\mathcal{B}_t}\nabla_\theta \ell(f_\theta(x_i), y_i)$$

The mini-batch gradient is an **unbiased estimator** of the full gradient, with variance
$\propto 1/B$. **Linear scaling rule:** if you multiply batch size by $k$, multiply the
learning rate by $k$ (up to a limit), plus a warmup.

## 5.4 Momentum and adaptive methods

**Momentum** — accumulate an exponentially-decaying velocity to damp oscillation across
narrow valleys:

$$v_t = \gamma v_{t-1} + \nabla_\theta J(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\,v_t$$

with $\gamma \approx 0.9$. Effective step is up to $1/(1-\gamma) = 10\times$ larger along
consistent directions.

**Nesterov accelerated gradient** — look ahead before computing the gradient:

$$v_t = \gamma v_{t-1} + \nabla_\theta J(\theta_t - \eta\gamma v_{t-1})$$

**AdaGrad** — per-parameter learning rates, scaled by accumulated squared gradients:

$$G_t = G_{t-1} + g_t^2,\qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon}g_t$$

Great for sparse features; but $G_t$ grows without bound so the learning rate decays to zero.

**RMSProp** — fix that with an exponential moving average:

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t}+\epsilon}g_t$$

**Adam** (Adaptive Moment Estimation) — momentum + RMSProp, with bias correction. The default
optimizer for deep learning:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad \text{(1st moment: mean)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \qquad \text{(2nd moment: uncentered variance)}$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t}+\epsilon}\hat m_t$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$
(use $10^{-4}$ to $3\times10^{-5}$ for transformers).

*Why bias correction?* $m_0 = 0$, so early estimates are biased toward zero; dividing by
$1-\beta_1^t$ (which $\to 1$) removes it.

**AdamW** — decouples weight decay from the adaptive scaling:

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_t\right)$$

In plain Adam, L2 regularization added to the loss gets divided by $\sqrt{\hat v}$, weakening
it for large-gradient parameters. AdamW applies decay directly. **Use AdamW.** Every modern
LLM does.

> **Memory cost:** Adam stores $m$ and $v$ per parameter. In fp32, a 7B model needs 28 GB for
> the weights + 56 GB for optimizer state + gradients. That's why full fine-tuning of large
> models needs multiple GPUs, and why LoRA exists.

## 5.5 Learning rate schedules

| Schedule | Formula |
|----------|---------|
| **Step decay** | $\eta_t = \eta_0\gamma^{\lfloor t/s\rfloor}$ |
| **Exponential** | $\eta_t = \eta_0 e^{-kt}$ |
| **Cosine annealing** | $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0-\eta_{\min})\left(1+\cos\frac{t\pi}{T}\right)$ |
| **Linear warmup** | $\eta_t = \eta_0 \cdot t/t_{\text{warmup}}$ for $t<t_{\text{warmup}}$ |
| **1cycle** | Warm up to a high peak, then anneal down |

**Warmup + cosine decay is the standard recipe for transformers.** Warmup matters because
Adam's second-moment estimate is unreliable in the first few hundred steps, and a large early
step can wreck the model.

## 5.6 Second-order and other methods

**Newton's method:** $\theta_{t+1} = \theta_t - H^{-1}\nabla J$. Quadratic convergence, but
$O(d^2)$ memory and $O(d^3)$ per step — impossible for deep nets.

**Quasi-Newton (BFGS/L-BFGS):** approximates $H^{-1}$ from gradient history. Excellent for
small/medium convex problems (`sklearn`'s default logistic regression solver is `lbfgs`).

**Coordinate descent:** optimize one coordinate at a time — the standard solver for lasso,
because the $L^1$ subproblem has a closed form (soft-thresholding).

**Gradient clipping** — prevents exploding gradients:

$$\mathbf{g} \leftarrow \mathbf{g}\cdot\min\left(1, \frac{c}{\|\mathbf{g}\|_2}\right)$$

Essential for RNNs and transformers; typical $c = 1.0$.

---

# Part 6 — Information Theory

**Why it matters:** cross-entropy is *the* classification loss; KL divergence powers VAEs,
RLHF/PPO, distillation, and drift detection.

## 6.1 Entropy

**Self-information** of an outcome: $I(x) = -\log p(x)$. Rare events carry more information.

**Entropy** — the expected information, i.e. average uncertainty:

$$H(X) = -\sum_x p(x)\log p(x) = \mathbb{E}[-\log p(X)]$$

Units: bits ($\log_2$) or nats ($\ln$). Maximal for a uniform distribution
($H = \log n$); zero for a deterministic one.

*Interpretation:* the minimum average number of bits needed to encode samples from $p$
(Shannon's source coding theorem).

## 6.2 Cross-entropy

The average bits needed to encode samples from $p$ using a code optimized for $q$:

$$H(p,q) = -\sum_x p(x)\log q(x)$$

**In ML** with $p$ = one-hot true labels and $q$ = model's predicted probabilities, this
collapses to the loss you use every day:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat y_k = -\log \hat y_{\text{correct class}}$$

Binary case:

$$\mathcal{L}_{\text{BCE}} = -\big[y\log\hat y + (1-y)\log(1-\hat y)\big]$$

## 6.3 KL divergence

**Kullback–Leibler divergence** — how much information is lost approximating $p$ with $q$:

$$D_{\text{KL}}(p\,\|\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p) \;\ge 0$$

$D_{\text{KL}} = 0$ iff $p = q$. **It is not symmetric** and is not a metric:
$D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p)$.

- **Forward KL** $D_{\text{KL}}(p\|q)$ is *mean-seeking* — $q$ must cover everywhere $p$ has mass.
- **Reverse KL** $D_{\text{KL}}(q\|p)$ is *mode-seeking* — $q$ collapses onto one mode. Used in
  variational inference.

Since $H(p)$ is fixed by the data, **minimizing cross-entropy ≡ minimizing KL divergence ≡
maximizing likelihood.** Three names for the same thing.

**Jensen–Shannon divergence** — the symmetric, bounded version:
$\text{JSD}(p\|q) = \tfrac12 D_{\text{KL}}(p\|m) + \tfrac12 D_{\text{KL}}(q\|m)$ with $m = \tfrac12(p+q)$.

**Where KL shows up:** the VAE's ELBO regularizer; PPO's constraint keeping the policy near the
reference model (RLHF); knowledge distillation's teacher–student loss; population-stability-index
style drift monitoring.

## 6.4 Mutual information

$$I(X;Y) = \sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = H(X) - H(X\mid Y) = D_{\text{KL}}\big(p(x,y)\,\|\,p(x)p(y)\big)$$

The reduction in uncertainty about $X$ from knowing $Y$. Zero iff independent. Unlike
correlation, it captures **non-linear** dependence.

**In ML:** feature selection (`mutual_info_classif`), and **information gain** in decision
trees is exactly mutual information between the split feature and the label.

## 6.5 Perplexity

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i)\right) = e^{H}$$

The standard language-model metric: the "effective vocabulary size" the model is choosing
among. PPL of 20 ≈ as uncertain as picking uniformly among 20 tokens. Lower is better.

---

# Part 7 — Numerical Computing

Mathematics that is correct on paper can still fail in floating point. These are the failures
you will actually hit.

## 7.1 Floating point

| Type | Bits | Approx. range | Use |
|------|------|--------------|-----|
| `float64` | 64 | $10^{\pm308}$ | Scientific computing, `sklearn` |
| `float32` | 32 | $10^{\pm38}$ | Default for deep learning |
| `float16` | 16 | $10^{\pm5}$ | Mixed precision (overflow-prone) |
| `bfloat16` | 16 | $10^{\pm38}$ | Same exponent range as fp32, fewer mantissa bits — **preferred for training** |
| `int8` / `fp8` / `int4` | 8/4 | — | Quantized inference |

**Machine epsilon** — the smallest $\epsilon$ with $1+\epsilon \ne 1$: about $2.2\times10^{-16}$
(fp64), $1.2\times10^{-7}$ (fp32). Consequences: `0.1 + 0.2 != 0.3`, and floating-point addition
is **not associative**, so GPU reductions are not bitwise reproducible across runs.

## 7.2 Numerical stability tricks

**Log-sum-exp** — computing $\log\sum_i e^{x_i}$ naively overflows. Subtract the max:

$$\log\sum_i e^{x_i} = c + \log\sum_i e^{x_i - c}, \qquad c = \max_i x_i$$

**Stable softmax** — the same trick, which is what every framework actually implements:

$$\text{softmax}(x)_i = \frac{e^{x_i - \max_j x_j}}{\sum_k e^{x_k - \max_j x_j}}$$

**Always use fused loss functions.** `nn.CrossEntropyLoss` takes *logits*, not probabilities,
because computing $\log(\text{softmax}(x))$ in one fused step avoids catastrophic cancellation.
Applying softmax then `log` yourself is a classic silent bug (and produces `-inf` when a
probability underflows to 0).

**Epsilon guards:** `x / (y + 1e-8)`, `log(p + 1e-8)`.

**Catastrophic cancellation:** subtracting near-equal numbers destroys precision. This is why
the "computational" variance formula $\mathbb{E}[X^2]-\mathbb{E}[X]^2$ is numerically worse
than the two-pass formula, and why Welford's online algorithm exists.

## 7.3 Vectorization

NumPy/PyTorch operations run in compiled C/CUDA; Python loops don't. The difference is
typically 10–100×.

```python
# ❌ 100× slower
result = [x[i] * y[i] for i in range(len(x))]

# ✅
result = x * y
```

**Broadcasting rules:** align shapes from the right; dimensions are compatible if equal or one
of them is 1; size-1 dimensions are stretched (without copying memory).

```
(3, 1) + (1, 4)  →  (3, 4)     ✅
(5, 3) + (3,)    →  (5, 3)     ✅   # (3,) is treated as (1,3)
(5, 3) + (5,)    →  ERROR      ❌   # align from the right: 3 vs 5
```

## 7.4 Complexity you should know

| Operation | Complexity |
|-----------|-----------|
| Matrix multiply $(m\times n)(n\times p)$ | $O(mnp)$ |
| Matrix inverse / solve, $n\times n$ | $O(n^3)$ |
| SVD, $m\times n$ | $O(\min(m^2n, mn^2))$ |
| Sorting $n$ items | $O(n\log n)$ |
| kNN query (brute force) | $O(nd)$ per query |
| Training a decision tree | $O(n d\log n)$ |
| Self-attention, sequence length $L$ | $O(L^2 d)$ ← why long context is expensive |
| Transformer FFN | $O(L d^2)$ |

---

## Self-check before moving on

You should be able to, without notes:

1. Explain why $\nabla f$ is the steepest-ascent direction.
2. State what $X^\top X$ singular means and how ridge fixes it.
3. Derive MSE from a Gaussian likelihood.
4. Explain why cross-entropy, KL divergence, and MLE are the same objective.
5. Write the Adam update from memory and say what $m_t$ and $v_t$ mean.
6. Explain the log-sum-exp trick and when you'd need it.
7. Say what the Hessian's eigenvalues tell you about a critical point.
8. Explain why halving your uncertainty needs 4× the data.

---

**Next:** [02 — ML Fundamentals →](02_ml_fundamentals.md)
