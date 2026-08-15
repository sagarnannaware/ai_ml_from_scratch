# 05 — Deep Learning

> From a single artificial neuron to the transformer, with backpropagation derived by hand.
> This is the module where "understanding" and "using a library" diverge most sharply.

---

## Table of contents

1. [The neuron and the perceptron](#1-the-neuron-and-the-perceptron)
2. [Multilayer perceptrons](#2-multilayer-perceptrons)
3. [Activation functions](#3-activation-functions)
4. [Backpropagation](#4-backpropagation)
5. [Optimization in deep learning](#5-optimization-in-deep-learning)
6. [Making deep networks trainable](#6-making-deep-networks-trainable)
7. [Convolutional Neural Networks (CNNs)](#7-convolutional-neural-networks-cnns)
8. [Recurrent Neural Networks](#8-recurrent-neural-networks)
9. [Attention and Transformers](#9-attention-and-transformers)
10. [Training in practice](#10-training-in-practice)
11. [Debugging neural networks](#11-debugging-neural-networks)

---

## 1. The neuron and the perceptron

### The artificial neuron

$$y = \phi\left(\sum_{j=1}^{d}w_jx_j + b\right) = \phi(\mathbf{w}^\top\mathbf{x}+b)$$

| Symbol | Name | Role |
|--------|------|------|
| $\mathbf{x}$ | Input | Features or previous layer's activations |
| $\mathbf{w}$ | Weights | Learned importance of each input |
| $b$ | Bias | Learned offset; shifts the activation threshold |
| $z = \mathbf{w}^\top\mathbf{x}+b$ | **Pre-activation / logit** | The linear part |
| $\phi$ | **Activation function** | The non-linearity |
| $y$ | Activation / output | What gets passed on |

### The perceptron (Rosenblatt, 1958)

$\phi$ = step function. Learning rule: for each misclassified example,
$\mathbf{w} \leftarrow \mathbf{w} + \eta(y - \hat y)\mathbf{x}$.

**Convergence theorem:** if the data is linearly separable, the perceptron converges in a
finite number of steps.

**The XOR problem (Minsky & Papert, 1969):** a single perceptron cannot represent XOR, because
XOR isn't linearly separable — no single line separates $\{(0,0),(1,1)\}$ from
$\{(0,1),(1,0)\}$. This finding triggered the first "AI winter."

**The resolution:** stack layers. A two-layer network solves XOR trivially, because the hidden
layer *learns a new representation* in which the problem becomes linearly separable. That is
the entire idea of deep learning in one sentence.

---

## 2. Multilayer Perceptrons

📄 `supervised/neural_network_classifier_example.py`

### Architecture

A stack of fully-connected (dense) layers. For layer $l$:

$$\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \phi^{(l)}(\mathbf{z}^{(l)})$$

with $\mathbf{a}^{(0)} = \mathbf{x}$ and the network output $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$.

**Shapes:** if layer $l$ has $n_l$ units, then $W^{(l)}\in\mathbb{R}^{n_l\times n_{l-1}}$ and
$\mathbf{b}^{(l)}\in\mathbb{R}^{n_l}$. Batched (rows = examples): $Z = XW^\top + \mathbf{b}$.

**Parameter count:** $\sum_l (n_{l-1}\cdot n_l + n_l)$.

### Universal Approximation Theorem

A feedforward network with a **single** hidden layer containing finitely many neurons and a
non-polynomial activation can approximate any continuous function on a compact subset of
$\mathbb{R}^n$ to arbitrary accuracy (Cybenko 1989; Hornik 1991).

**What it does NOT say:**
- How many neurons you need (possibly exponentially many).
- That you can *find* those weights by training.
- That the result generalizes.

**Why depth then?** Depth buys **exponential efficiency**. Certain functions requiring
$O(2^n)$ neurons in a shallow network need only $O(n)$ in a deep one. Deep networks compose
features hierarchically: edges → textures → parts → objects.

### Output layer design

| Task | Output units | Activation | Loss |
|------|-------------|-----------|------|
| Regression | 1 | None (linear) | MSE / Huber |
| Multi-output regression | $m$ | None | MSE |
| Binary classification | 1 | Sigmoid | BCE |
| Multiclass (single label) | $K$ | Softmax | Categorical CE |
| Multilabel | $K$ | Sigmoid (each independent) | Per-label BCE |

> **Pass logits, not probabilities, to your loss function.** `nn.CrossEntropyLoss` and
> `nn.BCEWithLogitsLoss` apply the activation internally in a numerically stable fused way.
> Applying softmax yourself and then `NLLLoss`-ing the log is a classic source of NaNs.

---

## 3. Activation Functions

Without non-linearity, stacked linear layers collapse: $W_2(W_1\mathbf{x}) = (W_2W_1)\mathbf{x}$
is just another linear map. **The activation is what makes depth meaningful.**

| Function | Formula | Derivative | Range | Notes |
|----------|---------|-----------|-------|-------|
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ | $(0,1)$ | Saturates; max gradient 0.25 → vanishing. Output layers only |
| **Tanh** | $\frac{e^z-e^{-z}}{e^z+e^{-z}}$ | $1-\tanh^2$ | $(-1,1)$ | Zero-centered (better than sigmoid); still saturates |
| **ReLU** | $\max(0,z)$ | $1$ if $z>0$ else $0$ | $[0,\infty)$ | **The default.** Fast, no saturation for $z>0$ |
| **Leaky ReLU** | $\max(\alpha z, z)$, $\alpha\approx0.01$ | $1$ or $\alpha$ | $\mathbb{R}$ | Fixes dying ReLU |
| **PReLU** | Leaky with learned $\alpha$ | — | $\mathbb{R}$ | Slightly better, more params |
| **ELU** | $z$ if $z>0$, else $\alpha(e^z-1)$ | — | $(-\alpha,\infty)$ | Smooth, zero-centered mean |
| **GELU** | $z\cdot\Phi(z)$ | — | $\mathbb{R}$ | **Standard in transformers** (BERT, GPT) |
| **SiLU/Swish** | $z\cdot\sigma(z)$ | — | $\mathbb{R}$ | Smooth; used in EfficientNet, Llama |
| **SwiGLU** | $\text{Swish}(zW)\odot(zV)$ | — | $\mathbb{R}$ | Gated; used in Llama, PaLM |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | see below | $(0,1)$, sums to 1 | Multiclass output only |

**GELU** (Gaussian Error Linear Unit): $\text{GELU}(z) = z\cdot P(Z\le z) = z\Phi(z)$, where
$\Phi$ is the standard normal CDF. It weights the input by the probability that a standard
normal falls below it — a smooth, probabilistic gate. Common approximation:

$$\text{GELU}(z)\approx 0.5z\left(1+\tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)$$

**Softmax Jacobian** (needed for the backprop derivation):

$$\frac{\partial\,\text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i(\delta_{ij} - \text{softmax}(z)_j)$$

### The dying ReLU problem
If a neuron's pre-activation is negative for every input, its gradient is 0 forever and it
never recovers — it's dead. Caused by too-large learning rates or a big negative bias.
Mitigate with Leaky ReLU/GELU, lower learning rates, and good initialization.

### Choosing
Hidden layers: **ReLU** by default; **GELU/SiLU** for transformers. Output layer: dictated by
the task (table above). Don't overthink it — the difference between modern activations is
typically under 1%.

---

## 4. Backpropagation

**The single most important algorithm in deep learning.** It is the chain rule, applied with
dynamic programming so that shared subexpressions are computed once.

### Setup: a 2-layer network

$$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \phi(\mathbf{z}^{(1)})$$
$$\mathbf{z}^{(2)} = W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}, \quad \hat{\mathbf{y}} = \text{softmax}(\mathbf{z}^{(2)})$$
$$L = -\sum_{k}y_k\log\hat y_k$$

### Step 1 — output layer error

Combining the cross-entropy derivative with the softmax Jacobian (the terms cancel elegantly,
as in [Module 03 §2](03_supervised_learning.md#gradient-derivation)):

$$\boxed{\boldsymbol\delta^{(2)} \equiv \frac{\partial L}{\partial\mathbf{z}^{(2)}} = \hat{\mathbf{y}} - \mathbf{y}}$$

### Step 2 — gradients for the output layer's parameters

Since $\mathbf{z}^{(2)} = W^{(2)}\mathbf{a}^{(1)}+\mathbf{b}^{(2)}$:

$$\frac{\partial L}{\partial W^{(2)}} = \boldsymbol\delta^{(2)}\,\mathbf{a}^{(1)\top}, \qquad \frac{\partial L}{\partial\mathbf{b}^{(2)}} = \boldsymbol\delta^{(2)}$$

Note the shape: $(n_2 \times 1)(1\times n_1) = (n_2\times n_1)$ ✓ matches $W^{(2)}$.

### Step 3 — propagate the error backwards

$$\frac{\partial L}{\partial\mathbf{a}^{(1)}} = W^{(2)\top}\boldsymbol\delta^{(2)}$$

$$\boxed{\boldsymbol\delta^{(1)} = \frac{\partial L}{\partial\mathbf{z}^{(1)}} = \left(W^{(2)\top}\boldsymbol\delta^{(2)}\right)\odot\phi'(\mathbf{z}^{(1)})}$$

where $\odot$ is the element-wise (Hadamard) product. **The error is passed back through the
transposed weights, then gated by the local activation derivative.**

### Step 4 — gradients for the first layer

$$\frac{\partial L}{\partial W^{(1)}} = \boldsymbol\delta^{(1)}\mathbf{x}^\top, \qquad \frac{\partial L}{\partial\mathbf{b}^{(1)}} = \boldsymbol\delta^{(1)}$$

### The general recursion

For any layer $l$ from $L-1$ down to 1:

$$\boldsymbol\delta^{(L)} = \nabla_{\mathbf{a}^{(L)}}L \odot \phi'(\mathbf{z}^{(L)})$$
$$\boldsymbol\delta^{(l)} = \left(W^{(l+1)\top}\boldsymbol\delta^{(l+1)}\right)\odot\phi'(\mathbf{z}^{(l)})$$
$$\frac{\partial L}{\partial W^{(l)}} = \boldsymbol\delta^{(l)}\mathbf{a}^{(l-1)\top}, \qquad \frac{\partial L}{\partial\mathbf{b}^{(l)}} = \boldsymbol\delta^{(l)}$$

**Four equations. That's all of backpropagation.**

### Why it's efficient

Naively computing each parameter's gradient separately costs $O(P^2)$ for $P$ parameters.
Backprop reuses $\boldsymbol\delta^{(l+1)}$ when computing $\boldsymbol\delta^{(l)}$, making
the backward pass cost about the same as the forward pass — $O(P)$. That's the whole trick:
**dynamic programming over the computational graph.**

### Reference implementation (pure NumPy)

```python
import numpy as np

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1.T + b1              # (n, h)
    A1 = np.maximum(0, Z1)          # ReLU
    Z2 = A1 @ W2.T + b2             # (n, K)
    Z2 = Z2 - Z2.max(axis=1, keepdims=True)      # stable softmax
    expZ = np.exp(Z2)
    Y_hat = expZ / expZ.sum(axis=1, keepdims=True)
    return Z1, A1, Y_hat

def backward(X, Y, Z1, A1, Y_hat, W2):
    n = X.shape[0]
    D2 = (Y_hat - Y) / n            # (n, K)   ← softmax + CE gradient
    dW2 = D2.T @ A1                 # (K, h)
    db2 = D2.sum(axis=0)
    D1 = (D2 @ W2) * (Z1 > 0)       # (n, h)   ← backprop through ReLU
    dW1 = D1.T @ X                  # (h, d)
    db1 = D1.sum(axis=0)
    return dW1, db1, dW2, db2
```

### Automatic differentiation

Frameworks build a **computational graph** of operations, then apply the chain rule mechanically.

| Mode | Cost | Best when |
|------|------|-----------|
| **Forward-mode** | One pass per *input* | Few inputs, many outputs |
| **Reverse-mode** (backprop) | One pass per *output* | **Many inputs, one output** — exactly ML's shape (millions of parameters → one scalar loss) |

**Gradient checking** — verify an implementation with finite differences:

$$\frac{\partial L}{\partial\theta_i} \approx \frac{L(\theta_i+\epsilon) - L(\theta_i-\epsilon)}{2\epsilon}, \qquad \epsilon\approx10^{-5}$$

Relative error below $10^{-7}$ is correct; above $10^{-4}$ means a bug. Use `float64`.

### Vanishing and exploding gradients

The gradient at layer 1 of an $L$-layer network is a product of $L$ terms:

$$\frac{\partial L}{\partial W^{(1)}} \propto \prod_{l=2}^{L}W^{(l)\top}\,\text{diag}(\phi'(\mathbf{z}^{(l)}))$$

- If the factors are consistently $<1$ → the product **decays exponentially** → early layers
  don't learn (**vanishing**). Sigmoid's max derivative of 0.25 makes this severe: $0.25^{10}\approx10^{-6}$.
- If consistently $>1$ → **explodes** → NaN losses.

Fixes: ReLU-family activations, careful initialization, normalization layers, **residual
connections**, and gradient clipping.

---

## 5. Optimization in Deep Learning

The full mathematics is in [Module 01 §5](01_math_foundations.md#part-5--optimization). Here's
what to actually do.

### The training loop

```python
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()               # 1. clear old gradients
        y_pred = model(X_batch)             # 2. forward pass
        loss = criterion(y_pred, y_batch)   # 3. compute loss
        loss.backward()                     # 4. backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 5. clip
        optimizer.step()                    # 6. update weights
        scheduler.step()                    # 7. update learning rate

    model.eval()                            # ← disables dropout, freezes BN stats
    with torch.no_grad():
        validate(model, val_loader)
```

> **`optimizer.zero_grad()` is mandatory.** PyTorch *accumulates* gradients by default (useful
> for gradient accumulation across micro-batches). Forgetting it silently sums gradients across
> batches and your model quietly fails to learn.

### Choosing an optimizer

| Situation | Use |
|-----------|-----|
| Default / unsure | **AdamW**, lr = $3\times10^{-4}$ |
| Transformers / LLMs | AdamW, lr = $10^{-4}$ to $3\times10^{-5}$, warmup + cosine decay |
| CNNs from scratch | SGD + momentum 0.9, lr = 0.1 with step decay — often generalizes *better* than Adam |
| Fine-tuning | AdamW, lr = $10^{-5}$ to $5\times10^{-5}$ (much smaller than from-scratch) |
| Sparse features | Adam / AdaGrad |

### Finding the learning rate

**LR range test:** train for a few hundred steps while exponentially increasing the learning
rate; plot loss vs lr. Pick the lr about one order of magnitude below where the loss starts
diverging (i.e. the steepest downward slope, not the minimum).

### Batch size

| Small (16–64) | Large (512–8192) |
|---------------|------------------|
| Noisier gradients — a regularizer | Smoother gradients |
| Often generalizes better | Faster wall-clock (better GPU utilization) |
| Less memory | Needs lr scaling + warmup |

**Gradient accumulation** simulates a large batch on small hardware:

```python
for i, (X, y) in enumerate(loader):
    loss = criterion(model(X), y) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 6. Making Deep Networks Trainable

Six ideas turned "deep networks don't train" into "deep networks are routine."

### 6.1 Weight initialization

Initializing all weights to zero makes every neuron in a layer compute the same thing forever
(**symmetry**), so gradients are identical and the layer collapses to one unit. Random init
breaks the symmetry — but the *scale* matters enormously.

**Xavier/Glorot** (for tanh/sigmoid) — keeps variance stable in both directions:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}+n_{\text{out}}}\right) \quad\text{or}\quad \mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right]$$

**He/Kaiming** (for ReLU) — ReLU zeroes half the activations, so the variance must be doubled
to compensate:

$$W\sim\mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

**Rule:** He init with ReLU, Xavier with tanh, and biases at 0. Frameworks default sensibly,
but if you write custom layers, initialize deliberately.

### 6.2 Batch Normalization

Normalize each feature across the batch, then let the network learn its own scale and shift:

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m(x_i-\mu_B)^2$$
$$\hat x_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\epsilon}}, \qquad y_i = \gamma\hat x_i + \beta$$

$\gamma$ and $\beta$ are **learned** — the network can undo the normalization if that's optimal.

**Train vs inference:** during training, batch statistics are used and a running average is
accumulated; at inference, the running averages are used (a batch of 1 has no meaningful
statistics). **This is why `model.eval()` matters.**

**Why it helps:** the original "internal covariate shift" explanation is now disputed. The
better-supported account is that BN **smooths the loss landscape**, allowing higher learning
rates and providing mild regularization through batch noise.

**Limitations:** depends on batch size (bad for small batches), awkward for RNNs/variable
lengths, adds train/inference asymmetry.

### 6.3 Layer Normalization

Normalize across the **features** of each individual example, not across the batch:

$$\mu = \frac{1}{d}\sum_{j=1}^{d}x_j, \qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2, \qquad y = \gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$$

Batch-size independent and identical at train and inference. **The standard in transformers.**

**RMSNorm** — a cheaper variant that drops the mean-centering (used in Llama):

$$y = \gamma\cdot\frac{x}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}}$$

| Norm | Normalizes over | Used in |
|------|----------------|---------|
| BatchNorm | Batch, per channel | CNNs |
| LayerNorm | Features, per example | **Transformers**, RNNs |
| RMSNorm | Features (no centering) | Llama, modern LLMs |
| GroupNorm | Channel groups, per example | Small-batch vision, detection |
| InstanceNorm | Per channel per example | Style transfer |

### 6.4 Residual connections

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

**Why this changed everything:** the gradient through the skip path is

$$\frac{\partial\mathbf{y}}{\partial\mathbf{x}} = \frac{\partial\mathcal{F}}{\partial\mathbf{x}} + I$$

The $+I$ means gradients have a **direct highway** to earlier layers regardless of what
$\mathcal{F}$ does. Even if $\partial\mathcal{F}/\partial\mathbf{x}\to0$, the gradient still
flows. This is what made 100+ layer networks (ResNet, 2015) and every modern transformer
trainable.

Also: the block only needs to learn a *residual* correction; if the identity is already
optimal, it can just output 0, which is easy.

### 6.5 Dropout
Covered in [Module 02 §6](02_ml_fundamentals.md#6-regularization). Typical rates: 0.5 for dense
layers, 0.1 for transformers, rarely with convolutions.

### 6.6 Pre-norm vs post-norm

Original transformer: $\mathbf{x} + \text{Sublayer}(\text{LN}(\mathbf{x}))$ is **pre-norm**;
$\text{LN}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$ is **post-norm**.

Post-norm (the 2017 original) needs careful warmup to be stable. **Pre-norm is far more stable
for deep stacks and is what all modern LLMs use.**

---

## 7. Convolutional Neural Networks (CNNs)

📄 See also [Module 07 — Computer Vision](07_computer_vision.md)

### The convolution operation

For a 2-D input $I$ and kernel $K$ of size $k\times k$:

$$(I * K)(i,j) = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1}I(i+m,\,j+n)\,K(m,n)$$

(Technically cross-correlation, since deep learning skips the kernel flip — nobody minds,
because the kernel is learned.)

With $C_{\text{in}}$ input channels and $C_{\text{out}}$ filters:

$$O(i,j,c_{\text{out}}) = \sum_{c=0}^{C_{\text{in}}-1}\sum_{m}\sum_{n} I(i+m,\,j+n,\,c)\,K(m,n,c,c_{\text{out}}) + b_{c_{\text{out}}}$$

### The three key ideas

| Property | Meaning | Benefit |
|----------|---------|---------|
| **Local connectivity** | Each output depends only on a small input patch | Matches the fact that nearby pixels are related |
| **Parameter sharing** | The same kernel slides across the whole image | Massive parameter reduction; **translation equivariance** |
| **Hierarchical composition** | Stacked layers grow the receptive field | Edges → textures → parts → objects |

**Parameter comparison:** a dense layer from a 224×224×3 image to 1000 units needs
150M parameters. A 3×3 conv with 64 filters needs $3\times3\times3\times64+64 = 1{,}792$.

### Output size arithmetic

$$O = \left\lfloor\frac{W - K + 2P}{S}\right\rfloor + 1$$

| Symbol | Meaning |
|--------|---------|
| $W$ | Input size | 
| $K$ | Kernel size |
| $P$ | Padding |
| $S$ | Stride |

**"Same" padding** (output size = input size, when $S=1$): $P = \frac{K-1}{2}$. So a 3×3 kernel
needs $P=1$, a 5×5 needs $P=2$.

**Parameters in a conv layer:** $(K\times K\times C_{\text{in}} + 1)\times C_{\text{out}}$.

**Dilated (atrous) convolution** inserts gaps of size $r$ in the kernel, expanding the receptive
field without extra parameters. Effective kernel size: $K + (K-1)(r-1)$.

**Receptive field** growth for stacked 3×3 convs: layer $n$ sees $1 + 2n$ pixels. Two stacked
3×3 convs have the same receptive field as one 5×5 but use fewer parameters ($2\times9=18$ vs
25) and add an extra non-linearity — which is why VGG used small kernels exclusively.

### Pooling

**Max pooling:** take the max in each window. **Average pooling:** take the mean.
**Global average pooling:** average each entire channel to one number — replaces the giant
flatten+dense layer at the end of a CNN (as in ResNet), cutting parameters dramatically.

Pooling provides local translation *invariance* and downsampling. Modern architectures
increasingly use strided convolutions instead.

### A canonical CNN

```
INPUT (224×224×3)
  → [CONV 3×3 → BN → ReLU] × 2 → MAXPOOL 2×2      # 112×112×64
  → [CONV 3×3 → BN → ReLU] × 2 → MAXPOOL 2×2      # 56×56×128
  → [CONV 3×3 → BN → ReLU] × 3 → MAXPOOL 2×2      # 28×28×256
  → GLOBAL AVG POOL                                # 256
  → DENSE → SOFTMAX                                # K classes
```

### Landmark architectures

| Year | Model | Key contribution |
|------|-------|-----------------|
| 1998 | **LeNet-5** | The first practical CNN (digit recognition) |
| 2012 | **AlexNet** | ReLU + dropout + GPUs; won ImageNet by a huge margin, starting the deep learning era |
| 2014 | **VGG** | Depth via stacked 3×3 convs; simple and uniform |
| 2014 | **GoogLeNet/Inception** | Parallel multi-scale branches; 1×1 convs for cheap channel reduction |
| 2015 | **ResNet** | **Residual connections** → 152 layers; still the standard backbone |
| 2017 | **DenseNet** | Every layer connects to every later layer |
| 2017 | **MobileNet** | Depthwise separable convolutions for mobile |
| 2019 | **EfficientNet** | Compound scaling of depth/width/resolution |
| 2020 | **Vision Transformer** | Attention replaces convolution entirely |
| 2022 | **ConvNeXt** | A CNN modernized with transformer design choices — competitive again |

**Depthwise separable convolution** (MobileNet's core trick): factor a standard convolution
into a per-channel spatial convolution (depthwise) followed by a 1×1 channel mixing
(pointwise). Cost drops from $K^2C_{\text{in}}C_{\text{out}}$ to
$K^2C_{\text{in}} + C_{\text{in}}C_{\text{out}}$ — roughly an 8–9× reduction for 3×3 kernels.

---

## 8. Recurrent Neural Networks

### Vanilla RNN

Process a sequence one step at a time, carrying a hidden state:

$$\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h)$$
$$\mathbf{y}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y$$

The **same weights** are used at every timestep (parameter sharing across time), which is what
lets an RNN handle variable-length sequences.

**Backpropagation Through Time (BPTT):** unroll the network across timesteps and backpropagate.
Cost grows with sequence length; **truncated BPTT** limits the unroll depth.

### Why vanilla RNNs fail

The gradient across $T$ steps contains $\prod_{t}W_{hh}^\top\text{diag}(\tanh')$. If the largest
eigenvalue (spectral radius) of $W_{hh}$ is $<1$, the gradient **vanishes exponentially** in
$T$; if $>1$, it **explodes**. In practice, vanilla RNNs cannot learn dependencies beyond about
10 timesteps.

### LSTM (Long Short-Term Memory)

Adds a **cell state** $\mathbf{c}_t$ — a memory highway with *additive* updates — and three
gates controlling information flow:

$$\mathbf{f}_t = \sigma(W_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f) \qquad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(W_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i) \qquad \text{(input gate)}$$
$$\tilde{\mathbf{c}}_t = \tanh(W_c[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_c) \qquad \text{(candidate)}$$
$$\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1} + \mathbf{i}_t\odot\tilde{\mathbf{c}}_t \qquad \text{(cell update)}$$
$$\mathbf{o}_t = \sigma(W_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o) \qquad \text{(output gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{c}_t) \qquad \text{(hidden state)}$$

**Why it solves vanishing gradients:** the cell update is *additive*, not multiplicative. When
the forget gate is near 1, $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}\approx1$ and the
gradient passes through unchanged — the same insight as residual connections, five years
earlier.

> **Practical tip:** initialize the forget-gate bias to 1. This makes the network remember by
> default, which speeds up early training substantially.

### GRU (Gated Recurrent Unit)

A simplification: two gates, no separate cell state, ~25% fewer parameters.

$$\mathbf{z}_t = \sigma(W_z[\mathbf{h}_{t-1},\mathbf{x}_t]) \qquad \text{(update gate)}$$
$$\mathbf{r}_t = \sigma(W_r[\mathbf{h}_{t-1},\mathbf{x}_t]) \qquad \text{(reset gate)}$$
$$\tilde{\mathbf{h}}_t = \tanh(W[\mathbf{r}_t\odot\mathbf{h}_{t-1},\mathbf{x}_t])$$
$$\mathbf{h}_t = (1-\mathbf{z}_t)\odot\mathbf{h}_{t-1} + \mathbf{z}_t\odot\tilde{\mathbf{h}}_t$$

GRU vs LSTM: comparable accuracy, GRU is faster. Try GRU first.

### Seq2seq and the bottleneck that led to attention

Encoder–decoder: an encoder RNN compresses the input into a single fixed-size context vector;
a decoder RNN generates the output from it.

**The fatal flaw:** one vector must carry an entire sentence. Performance collapses on long
inputs. Bahdanau et al. (2014) fixed this by letting the decoder *attend* to all encoder states
— and that idea, taken to its conclusion, became the transformer.

**RNNs today** are largely superseded by transformers for NLP. They remain reasonable for small
time-series problems, streaming/online inference with strict memory limits, and very long
sequences where $O(L^2)$ attention is infeasible.

---

## 9. Attention and Transformers

*"Attention Is All You Need"* (Vaswani et al., 2017) — the architecture behind essentially all
frontier AI systems.

### The attention intuition

For each position, look at every other position and compute a weighted average of their values,
where the weights reflect relevance. It's a **soft, differentiable dictionary lookup**.

- **Query (Q):** what I'm looking for.
- **Key (K):** what I contain, advertised for matching.
- **Value (V):** what I actually give you if matched.

### Scaled dot-product attention

$$\boxed{\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}$$

Shapes: $Q\in\mathbb{R}^{L_q\times d_k}$, $K\in\mathbb{R}^{L_k\times d_k}$,
$V\in\mathbb{R}^{L_k\times d_v}$; output $\in\mathbb{R}^{L_q\times d_v}$.

Step by step:
1. $QK^\top$ → an $L_q\times L_k$ matrix of similarity scores (dot products).
2. Divide by $\sqrt{d_k}$ → **scaling**.
3. Softmax over the last axis → each row is a probability distribution over positions.
4. Multiply by $V$ → weighted average of the values.

**Why $\sqrt{d_k}$?** If the components of $q$ and $k$ are independent with mean 0 and variance
1, then $q^\top k = \sum_{i=1}^{d_k}q_ik_i$ has variance $d_k$ and standard deviation
$\sqrt{d_k}$. For $d_k = 64$ that's a spread of ±8 or more — pushed through softmax, the
distribution becomes nearly one-hot, and **the softmax gradient vanishes**
($\partial\text{softmax}$ is proportional to $p(1-p)\to0$). Dividing by $\sqrt{d_k}$ restores
unit variance and keeps gradients healthy. Without it, deep transformers don't train.

### Self-attention

$Q$, $K$, and $V$ are all linear projections of the *same* input $X$:

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

Every position attends to every other position — including itself. Crucially, the path length
between any two positions is **1**, versus $O(L)$ in an RNN. That's why transformers learn
long-range dependencies so much better.

### Multi-head attention

Run $h$ attention operations in parallel with different learned projections, then concatenate:

$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$

with $d_k = d_v = d_{\text{model}}/h$, so the total cost matches single-head attention at full
width. Different heads specialize — some track syntax, some coreference, some positional
patterns.

### Masking

**Causal (autoregressive) mask** — for decoders/GPT, position $i$ may only attend to positions
$\le i$. Implemented by setting the upper-triangular entries of the score matrix to $-\infty$
*before* the softmax (so they become exactly 0 after it):

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
scores = scores.masked_fill(causal_mask == 0, float('-inf'))
attn = scores.softmax(dim=-1) @ V
```

**Padding mask** — ignore padding tokens in variable-length batches.

### Positional encoding

Attention is **permutation-equivariant** — it has no inherent notion of order. Position must be
injected explicitly.

**Sinusoidal (original):**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Different dimensions oscillate at different frequencies, giving a unique multi-scale signature
per position, and relative positions are expressible as linear functions of these encodings.

| Scheme | Used in | Notes |
|--------|---------|-------|
| Sinusoidal | Original transformer | Extrapolates to unseen lengths |
| Learned absolute | BERT, GPT-2 | Simple; capped at trained length |
| **RoPE** (rotary) | **Llama, most modern LLMs** | Rotates Q and K by a position-dependent angle; encodes *relative* position naturally, extends well |
| ALiBi | BLOOM | Linear distance penalty on attention scores; strong length extrapolation |

### The full transformer block

```
x → LayerNorm → Multi-Head Self-Attention → + x     (residual)
  → LayerNorm → Feed-Forward Network       → + x     (residual)
```

**Feed-forward network** (applied identically at each position):

$$\text{FFN}(\mathbf{x}) = W_2\,\phi(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$$

with an inner dimension of $4d_{\text{model}}$ by convention. This holds roughly **two-thirds
of the model's parameters** and is where much of the factual knowledge appears to be stored.

### Architecture families

| Family | Attention | Trained by | Best for | Examples |
|--------|-----------|-----------|----------|----------|
| **Encoder-only** | Bidirectional | Masked LM | Understanding, classification, embeddings | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | Causal | Next-token prediction | **Generation** | GPT, Llama, Claude, Mistral |
| **Encoder–decoder** | Both + cross-attention | Denoising / seq2seq | Translation, summarization | T5, BART, original transformer |

**Cross-attention** (in encoder–decoder models): $Q$ comes from the decoder, $K$ and $V$ from
the encoder — this is how the decoder reads the source sequence.

### Complexity and the long-context problem

| Component | Time | Memory |
|-----------|------|--------|
| Self-attention | $O(L^2 d)$ | $O(L^2)$ for the score matrix |
| FFN | $O(Ld^2)$ | $O(Ld)$ |

The $O(L^2)$ term is the barrier to long context. Mitigations:

| Approach | Idea |
|----------|------|
| **FlashAttention** | Exact attention, but tiled to stay in SRAM — never materializes the $L\times L$ matrix. 2–4× faster, far less memory. **The standard now.** |
| **Sparse attention** | Attend to a subset (local windows + a few global tokens) — Longformer, BigBird |
| **Sliding window** | Each token attends to the last $w$ tokens — Mistral |
| **Linear attention** | Kernel-approximate softmax → $O(L)$ — Performer, Linformer |
| **Multi-Query / Grouped-Query Attention** | Heads share K/V projections → much smaller KV cache at inference |
| **State-space models** | Mamba/S4 — recurrent formulation with $O(L)$ scaling |

### Minimal implementation

```python
import torch, torch.nn as nn, math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k, self.h = d_model // n_heads, n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        # project, then split into heads: (B, h, L, d_k)
        q = self.W_q(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.h, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)   # (B, h, L, L)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        out = scores.softmax(dim=-1) @ v                            # (B, h, L, d_k)

        out = out.transpose(1, 2).contiguous().view(B, L, D)        # merge heads
        return self.W_o(out)
```

---

## 10. Training in Practice

### A sane recipe

1. **Overfit a single batch first.** If your model can't reach ~0 loss on 8 examples, you have a
   bug — not a modelling problem. This is the highest-value 5 minutes in deep learning.
2. Start with a known-good architecture and hyperparameters. Don't invent.
3. Add complexity one change at a time, measuring each.
4. Use mixed precision (`torch.cuda.amp` / `bf16`) — ~2× speedup, half the memory.
5. Log everything: loss curves, learning rate, gradient norms, weight norms, sample predictions.
6. Checkpoint every epoch; keep the best by validation metric.
7. Set seeds for reproducibility (accepting that GPU nondeterminism means it's approximate).

### Transfer learning

Almost never train from scratch. Start from a pretrained model:

| Strategy | When | How |
|----------|------|-----|
| **Feature extraction** | Very little data (<1k), similar domain | Freeze the backbone, train only a new head |
| **Fine-tune last layers** | Moderate data | Unfreeze the top few blocks |
| **Full fine-tuning** | Lots of data (>10k), or a domain shift | Unfreeze everything, use a small lr |
| **Discriminative lrs** | Any | Lower lr for early layers, higher for later ones |
| **PEFT / LoRA** | Large models, limited compute | Train small adapters ([Module 06](06_nlp_and_llms.md#6-adapting-a-pretrained-model)) |

### Data augmentation

| Domain | Techniques |
|--------|-----------|
| **Images** | Flip, rotate, crop, colour jitter, RandAugment, Mixup, CutMix, random erasing |
| **Text** | Back-translation, synonym replacement, token dropout, LLM paraphrasing |
| **Audio** | Time/frequency masking (SpecAugment), noise, pitch/speed shift |
| **Tabular** | SMOTE, Gaussian noise, feature dropout |

**Mixup:** train on convex combinations of examples *and* their labels:

$$\tilde{\mathbf{x}} = \lambda\mathbf{x}_i + (1-\lambda)\mathbf{x}_j, \qquad \tilde{\mathbf{y}} = \lambda\mathbf{y}_i+(1-\lambda)\mathbf{y}_j, \qquad \lambda\sim\text{Beta}(\alpha,\alpha)$$

### Distributed training

| Strategy | Splits | Use when |
|----------|--------|----------|
| **Data parallel (DDP)** | The batch across GPUs; gradients all-reduced | The model fits on one GPU |
| **Model/tensor parallel** | Individual layers across GPUs | A single layer is too large |
| **Pipeline parallel** | Layer groups across GPUs | A deep model doesn't fit |
| **FSDP / ZeRO** | Parameters, gradients, and optimizer state | Very large models — the standard for LLM training |

**Memory budget for training** (per parameter, mixed precision with Adam): ~2 bytes weights +
2 bytes gradients + 4 bytes fp32 master weights + 8 bytes Adam state ≈ **16 bytes/parameter**,
plus activations. A 7B model needs ~112 GB before activations — hence FSDP, LoRA, and gradient
checkpointing (recompute activations in the backward pass to trade compute for memory).

---

## 11. Debugging Neural Networks

| Symptom | Likely causes | What to check |
|---------|--------------|---------------|
| **Loss is NaN** | lr too high; log(0); division by 0; fp16 overflow | Lower lr, add epsilons, use fused losses, switch to bf16, clip gradients |
| **Loss doesn't decrease** | Forgot `zero_grad()`; lr too small; frozen params; wrong loss | Print gradient norms; verify `requires_grad`; overfit one batch |
| **Loss decreases then explodes** | lr too high; no clipping | Lower lr, add warmup, clip gradients |
| **Train good, val bad** | Overfitting; leakage; distribution mismatch | More regularization/data; audit the split |
| **Train and val both bad** | Underfitting; broken data pipeline | Bigger model, train longer, **visualize your actual inputs** |
| **Val better than train** | Dropout/BN active only in training; val set is easier | Normal to a small degree; check `model.eval()` |
| **Works in training, bad at inference** | Forgot `model.eval()`; preprocessing mismatch | The #1 production bug — check both |
| **Loss stuck at $\ln(K)$** | Model predicting uniform; dead network | Check init, activations, lr |
| **GPU OOM** | Batch too large; memory leak from retained graph | Reduce batch; use accumulation; `loss.item()` not `loss` when logging |

**Universal first move: overfit a single batch.** If it can't, the bug is in your code, not
your hyperparameters. Second move: **look at your data** — plot the actual tensors going into
the model. A shocking share of "modelling problems" are mislabeled data, a broken normalization,
or channels in the wrong order.

---

## Self-check

1. Write the four backpropagation equations from memory.
2. Explain why $\boldsymbol\delta$ is multiplied by $\phi'(\mathbf{z})$ element-wise.
3. Why does He init use $2/n_{\text{in}}$ while Xavier uses $2/(n_{\text{in}}+n_{\text{out}})$?
4. Explain why residual connections fix vanishing gradients, in terms of the derivative.
5. Derive why attention scales by $\sqrt{d_k}$ and describe what breaks without it.
6. Compute the output size of a 3×3 conv, stride 2, padding 1, on a 224×224 input.
7. Why is the LSTM cell update additive, and what does that buy you?
8. How much GPU memory does training a 7B model with AdamW in mixed precision need, and why?

---

**Next:** [06 — NLP & Large Language Models →](06_nlp_and_llms.md)
