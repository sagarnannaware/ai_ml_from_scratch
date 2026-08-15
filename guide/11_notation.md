# 11 — Notation Reference

> Every symbol used in this guide, with its meaning and typical shape. Mathematical notation in
> ML is inconsistent across textbooks; this is the convention used consistently throughout
> these modules.

---

## Conventions at a glance

| Style | Means | Example |
|-------|-------|---------|
| Lowercase italic | Scalar | $a$, $n$, $\lambda$ |
| Lowercase **bold** | Vector (column) | $\mathbf{x}$, $\mathbf{w}$ |
| Uppercase italic | Matrix | $X$, $W$, $A$ |
| Calligraphic | Set or function space | $\mathcal{D}$, $\mathcal{S}$, $\mathcal{L}$ |
| Hat | Estimate or prediction | $\hat y$, $\hat\theta$ |
| Bar | Sample mean | $\bar x$ |
| Star | Optimal value | $\theta^*$, $Q^*$ |
| Subscript $i$ | Index over examples | $\mathbf{x}_i$ |
| Subscript $j$ | Index over features | $x_j$ |
| Superscript $(l)$ | Layer index | $W^{(l)}$ |

---

## Data and indices

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $n$ | Number of training examples | scalar |
| $d$ | Number of features (input dimension) | scalar |
| $K$ | Number of classes | scalar |
| $B$ or $m$ | Batch size | scalar |
| $L$ | Sequence length (or number of layers, in context) | scalar |
| $\mathbf{x}$, $\mathbf{x}_i$ | A single input example | $(d,)$ |
| $x_j$ | Feature $j$ of an example | scalar |
| $X$ | Design matrix (all examples) | $(n, d)$ |
| $y$, $y_i$ | True label/target | scalar or $(K,)$ one-hot |
| $\mathbf{y}$ | Vector of all targets | $(n,)$ |
| $\hat y$, $\hat{\mathbf{y}}$ | Model prediction | matches $y$ |
| $\mathcal{D}$ | The dataset $\{(\mathbf{x}_i,y_i)\}_{i=1}^n$ | — |
| $\mathcal{X}$, $\mathcal{Y}$ | Input and output spaces | — |
| $\mathbb{R}^d$ | $d$-dimensional real space | — |

## Model and parameters

| Symbol | Meaning |
|--------|---------|
| $\theta$ | All model parameters, collectively |
| $\mathbf{w}$ | Weight vector (linear models) |
| $W$, $W^{(l)}$ | Weight matrix, layer $l$ |
| $b$, $\mathbf{b}$ | Bias term / vector |
| $f$, $f_\theta$ | The model function |
| $\pi_\theta$ | Policy (RL) |
| $\mathcal{H}$ | Hypothesis space — the set of functions the model can express |
| $z$, $\mathbf{z}^{(l)}$ | Pre-activation (logits) |
| $a$, $\mathbf{a}^{(l)}$ | Post-activation output of layer $l$ |
| $\phi$ | Activation function (also a feature map, in kernel methods) |
| $\sigma$ | Sigmoid function (also standard deviation — disambiguate by context) |
| $n_l$ | Number of units in layer $l$ |

## Loss, optimization, training

| Symbol | Meaning |
|--------|---------|
| $\ell$ | Per-example loss |
| $\mathcal{L}$, $J$ | Total objective / cost function |
| $R(f)$ | True risk — expected loss over the data distribution |
| $\hat R(f)$ | Empirical risk — average loss on the training set |
| $\Omega(\theta)$ | Regularization term |
| $\lambda$ | Regularization strength |
| $\eta$, $\alpha$ | Learning rate / step size |
| $\nabla_\theta$ | Gradient with respect to $\theta$ |
| $\nabla^2$, $H$ | Hessian (second derivatives) |
| $J$ (matrix context) | Jacobian |
| $\delta^{(l)}$ | Backpropagated error signal at layer $l$ |
| $g_t$ | Gradient at step $t$ |
| $m_t$, $v_t$ | Adam's first and second moment estimates |
| $\beta_1$, $\beta_2$ | Adam's decay rates (0.9, 0.999) |
| $\gamma$ | Momentum coefficient (also the RL discount factor) |
| $\epsilon$ | Small constant for numerical stability ($10^{-8}$) |
| $\odot$ | Element-wise (Hadamard) product |
| $*$ | Convolution |

## Probability and statistics

| Symbol | Meaning |
|--------|---------|
| $P(A)$ | Probability of event $A$ |
| $p(x)$ | Probability density/mass function |
| $P(A\mid B)$ | Conditional probability |
| $X \sim \mathcal{D}$ | $X$ is distributed according to $\mathcal{D}$ |
| $\mathbb{E}[X]$ | Expectation |
| $\text{Var}(X)$ | Variance |
| $\text{Cov}(X,Y)$ | Covariance |
| $\Sigma$ | Covariance matrix |
| $\rho$ | Correlation coefficient |
| $\mu$, $\boldsymbol\mu$ | Mean (scalar / vector) |
| $\sigma$, $\sigma^2$ | Standard deviation, variance |
| $\mathcal{N}(\mu,\sigma^2)$ | Normal (Gaussian) distribution |
| $\mathcal{U}(a,b)$ | Uniform distribution |
| $\text{Bern}(p)$ | Bernoulli distribution |
| $\perp$ | Independence: $X \perp Y$ |
| $\xrightarrow{d}$ | Convergence in distribution |
| $\hat\theta$ | An estimator of $\theta$ |
| $H_0$, $H_1$ | Null and alternative hypotheses |
| $\alpha$ (testing) | Significance level |

## Linear algebra

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}^\top$, $A^\top$ | Transpose |
| $A^{-1}$ | Matrix inverse |
| $A^{+}$ | Moore–Penrose pseudo-inverse |
| $\det(A)$ | Determinant |
| $\text{tr}(A)$ | Trace |
| $\text{rank}(A)$ | Rank |
| $I$ | Identity matrix |
| $\|\cdot\|_p$ | $L^p$ norm |
| $\|\cdot\|_F$ | Frobenius norm |
| $\langle\mathbf{x},\mathbf{y}\rangle$ | Inner (dot) product |
| $\lambda_i$ | $i$-th eigenvalue |
| $\mathbf{v}_i$ | $i$-th eigenvector |
| $\sigma_i$ | $i$-th singular value |
| $U\Sigma V^\top$ | Singular value decomposition |
| $\succeq 0$ | Positive semi-definite |
| $\kappa$ | Condition number |

## Information theory

| Symbol | Meaning |
|--------|---------|
| $H(X)$ | Entropy |
| $H(p,q)$ | Cross-entropy |
| $H(X\mid Y)$ | Conditional entropy |
| $D_{\text{KL}}(p\|q)$ | Kullback–Leibler divergence |
| $I(X;Y)$ | Mutual information |
| $\text{PPL}$ | Perplexity |

## Reinforcement learning

| Symbol | Meaning |
|--------|---------|
| $s$, $s_t$ | State at time $t$ |
| $a$, $a_t$ | Action |
| $r$, $r_t$ | Reward |
| $\mathcal{S}$, $\mathcal{A}$ | State and action spaces |
| $P(s'\mid s,a)$ | Transition probability |
| $R(s,a)$ | Expected reward |
| $\gamma$ | Discount factor |
| $G_t$ | Return (cumulative discounted reward) |
| $\pi(a\mid s)$ | Policy |
| $V^\pi(s)$ | State-value function |
| $Q^\pi(s,a)$ | Action-value function |
| $A^\pi(s,a)$ | Advantage function |
| $V^*$, $Q^*$, $\pi^*$ | Optimal value functions and policy |
| $\delta_t$ | TD error |
| $\varepsilon$ | Exploration rate (ε-greedy) |
| $\tau$ | A trajectory (also softmax temperature) |
| $\lambda$ (GAE) | GAE bias–variance parameter |

## Transformers

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $Q$, $K$, $V$ | Query, Key, Value matrices | $(L, d_k)$ / $(L, d_v)$ |
| $d_{\text{model}}$ | Model/embedding dimension | scalar |
| $d_k$, $d_v$ | Key and value dimensions per head | scalar |
| $h$ | Number of attention heads | scalar |
| $W^Q, W^K, W^V, W^O$ | Attention projection matrices | $(d_{\text{model}}, d_{\text{model}})$ |
| $L$ | Sequence length | scalar |
| $PE$ | Positional encoding | $(L, d_{\text{model}})$ |

## Computer vision

| Symbol | Meaning |
|--------|---------|
| $H$, $W$ | Image height and width |
| $C$, $C_{\text{in}}$, $C_{\text{out}}$ | Channel counts |
| $K$ | Kernel size |
| $S$ | Stride |
| $P$ | Padding |
| $N$ | Batch size |
| $\bar\alpha_t$ | Cumulative noise schedule product (diffusion) |
| $\boldsymbol\epsilon_\theta$ | Noise prediction network (diffusion) |

---

## Common ambiguities to watch for

These symbols are overloaded across the field. Context disambiguates, but be alert:

| Symbol | Could mean |
|--------|-----------|
| $\alpha$ | Learning rate; significance level; Dirichlet/Beta parameter; LoRA scaling; focal loss weight; Lagrange multiplier |
| $\lambda$ | Regularization strength; eigenvalue; Poisson rate; GAE parameter; TD(λ) |
| $\gamma$ | RL discount; momentum; RBF kernel width; BatchNorm scale; focal loss exponent |
| $\sigma$ | Sigmoid function; standard deviation; singular value |
| $J$ | Objective function; Jacobian matrix |
| $L$ | Loss; sequence length; number of layers; likelihood |
| $K$ | Number of classes; number of clusters; kernel function; Key matrix; kernel size |
| $\epsilon$ | Numerical stability constant; PPO clip range; ε-greedy rate; noise variable; DBSCAN radius |
| $\tau$ | Trajectory; temperature; quantile level |
| $\pi$ | Policy; the constant 3.14159…; mixture weight |
| $H$ | Entropy; Hessian; image height; hat matrix |

---

## Reading mathematical ML papers

A practical decoding procedure when you hit unfamiliar notation:

1. **Find the shape first.** If you know a symbol is $(n,d)$, half the meaning follows.
2. **Check the index ranges.** $\sum_{i=1}^n$ (over examples) vs $\sum_{j=1}^d$ (over features)
   vs $\sum_{k=1}^K$ (over classes) tells you what's being aggregated.
3. **Identify what's learned vs fixed.** Anything with a $\theta$/$\phi$ subscript is learned.
4. **Look for the expectation.** $\mathbb{E}_{x\sim p}$ tells you what distribution the whole
   expression lives under — this is usually where the real assumption hides.
5. **Track the argmin/argmax variable.** $\arg\min_\theta$ says what's being optimized; every
   other symbol is held fixed.
6. **Reduce to one example.** Drop the sum over $i$ and the batch dimension, and most equations
   become obvious.

---

**Next:** [12 — Projects, Papers & Resources →](12_projects_and_resources.md)
