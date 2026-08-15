# 07 — Computer Vision

> Teaching machines to see: classification, detection, segmentation, vision transformers, and
> generative image models.

📄 Companion code: `python/07_opencv_computer_vision.py`

**Prerequisite:** [Module 05 §7 — CNNs](05_deep_learning.md#7-convolutional-neural-networks-cnns)
covers convolution arithmetic, which this module assumes.

---

## Table of contents

1. [Image fundamentals](#1-image-fundamentals)
2. [Classical computer vision](#2-classical-computer-vision)
3. [Image classification](#3-image-classification)
4. [Object detection](#4-object-detection)
5. [Segmentation](#5-segmentation)
6. [Vision Transformers](#6-vision-transformers)
7. [Self-supervised and multimodal vision](#7-self-supervised-and-multimodal-vision)
8. [Generative models](#8-generative-models)
9. [Practical vision engineering](#9-practical-vision-engineering)

---

## 1. Image Fundamentals

### Representation

A digital image is a tensor of pixel intensities.

| Type | Shape | Values |
|------|-------|--------|
| Grayscale | $(H, W)$ | 0–255 (uint8) or 0–1 (float) |
| RGB colour | $(H, W, 3)$ | 3 channels |
| RGBA | $(H, W, 4)$ | + alpha (transparency) |
| Batch (PyTorch) | $(N, C, H, W)$ | **channels-first** |
| Batch (TensorFlow) | $(N, H, W, C)$ | **channels-last** |

> **Channel order is a perennial bug source.** PyTorch is NCHW; TensorFlow/OpenCV are NHWC.
> And **OpenCV reads images as BGR, not RGB** — if your model's colours look wrong, that's why.

### Colour spaces

| Space | Channels | Use |
|-------|----------|-----|
| **RGB** | Red, Green, Blue | Display, default for neural nets |
| **BGR** | Reversed RGB | OpenCV's default |
| **HSV** | Hue, Saturation, Value | Colour-based thresholding — hue is illumination-robust |
| **LAB** | Lightness, a, b | Perceptually uniform; colour-difference measurement |
| **Grayscale** | Intensity | $Y = 0.299R + 0.587G + 0.114B$ (weighted by human luminance perception) |

### Preprocessing

**Normalization** — the standard for ImageNet-pretrained models:

$$x' = \frac{x/255 - \boldsymbol\mu}{\boldsymbol\sigma}, \qquad \boldsymbol\mu = [0.485, 0.456, 0.406],\; \boldsymbol\sigma = [0.229, 0.224, 0.225]$$

**You must use the same normalization the model was pretrained with.** Mismatched
normalization silently degrades accuracy.

**Resizing:** interpolation choice matters — bilinear (default), bicubic (sharper),
nearest-neighbour (**required for segmentation masks**, since interpolating class indices
creates nonsense labels), Lanczos (best downsampling quality).

---

## 2. Classical Computer Vision

Still relevant for preprocessing, industrial inspection, real-time constrained systems, and
when you have almost no data.

### Filtering and convolution

A **kernel** slides over the image computing a weighted sum (see [Module 05 §7](05_deep_learning.md#the-convolution-operation)).

| Kernel | Effect |
|--------|--------|
| **Box blur** | $\frac{1}{9}\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix}$ — simple smoothing |
| **Gaussian blur** | Weights by $G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$ — smooth, no ringing |
| **Sharpen** | $\begin{bmatrix}0&-1&0\\-1&5&-1\\0&-1&0\end{bmatrix}$ |
| **Sobel-x** | $\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$ — vertical edges |
| **Laplacian** | $\begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$ — second derivative, all edges |

**Median filter** (non-linear): replaces each pixel with the neighbourhood median — the best
choice for salt-and-pepper noise, and it preserves edges where Gaussian blur smears them.

**Bilateral filter:** Gaussian in both space *and* intensity — smooths flat regions while
keeping edges sharp.

### Edge detection

**Gradient magnitude and direction:**

$$G = \sqrt{G_x^2+G_y^2}, \qquad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$

**Canny edge detector** — still the gold standard:
1. Gaussian blur (suppress noise).
2. Compute gradients (Sobel).
3. **Non-maximum suppression** — thin edges to one pixel by keeping only local maxima along the
   gradient direction.
4. **Double thresholding** — strong edges (> high), weak edges (between), discard the rest.
5. **Hysteresis** — keep weak edges only if connected to a strong edge.

### Feature detectors and descriptors

| Method | Detects | Properties |
|--------|---------|-----------|
| **Harris corner** | Corners via the structure tensor's eigenvalues | Rotation-invariant, not scale-invariant |
| **SIFT** | Scale-Invariant Feature Transform | Scale + rotation invariant; 128-D descriptor; the classic |
| **SURF** | Faster SIFT approximation | Speed |
| **ORB** | Oriented FAST + rotated BRIEF | Fast, binary, free of patents — OpenCV's practical default |
| **HOG** | Histogram of Oriented Gradients | Pedestrian detection (pre-deep-learning SOTA) |

**Uses today:** image stitching/panoramas, SLAM and visual odometry, template matching,
camera calibration, and homography estimation — tasks where geometry matters more than
semantics.

### Morphological operations

On binary images, with a structuring element:

| Operation | Effect |
|-----------|--------|
| **Erosion** | Shrinks white regions; removes small noise |
| **Dilation** | Grows white regions; fills small holes |
| **Opening** (erode→dilate) | Removes small objects, preserves overall size |
| **Closing** (dilate→erode) | Fills small holes, preserves overall size |
| **Gradient** (dilate−erode) | Outlines |

**Thresholding:** global, **Otsu's method** (automatically picks the threshold minimizing
intra-class variance), or adaptive (per-region — necessary under uneven lighting).

---

## 3. Image Classification

Assign one label to an entire image.

### The standard approach: transfer learning

Training from scratch is almost never right. Start from an ImageNet-pretrained backbone:

```python
import torch.nn as nn
from torchvision import models

model = models.resnet50(weights="IMAGENET1K_V2")
for p in model.parameters():           # freeze the backbone
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)   # new head, trainable
```

| Data size | Strategy |
|-----------|----------|
| < 1k images | Freeze the backbone, train the head only |
| 1k–10k | Fine-tune the last block or two, low lr |
| > 10k | Fine-tune everything, lr ≈ $10^{-4}$ |
| Domain very different (medical, satellite) | Fine-tune everything, more epochs; ImageNet features still help |

### Augmentation

| Transform | Notes |
|-----------|-------|
| Random resized crop | The single most effective augmentation |
| Horizontal flip | Safe for most natural images — **not** for text, digits, or medical laterality |
| Colour jitter | Brightness/contrast/saturation/hue |
| Rotation, affine | Watch for edge artifacts |
| **RandAugment / AutoAugment** | Learned/randomized policies — strong defaults |
| **Mixup / CutMix** | Blend images and labels; excellent regularizers |
| Random erasing | Occlusion robustness |

**Only augment the training set.** Validation and test use deterministic resize + center crop.
**Test-time augmentation (TTA)** — averaging predictions over several augmented views — is a
cheap 0.5–1% accuracy gain at inference cost.

### Metrics
Top-1 and top-5 accuracy, per-class precision/recall, and a confusion matrix. For imbalanced
data, macro-F1 or balanced accuracy (see [Module 02 §7](02_ml_fundamentals.md#7-evaluation-metrics)).

---

## 4. Object Detection

Locate *and* classify multiple objects: output a set of (bounding box, class, confidence).

### Bounding box formats

| Format | Encoding | Used by |
|--------|----------|---------|
| `xyxy` (Pascal VOC) | $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ | torchvision |
| `xywh` (COCO) | $(x_{\min}, y_{\min}, w, h)$ | COCO |
| `cxcywh` (YOLO) | centre $(x, y, w, h)$, **normalized to [0,1]** | YOLO |

Converting between these incorrectly is the most common detection bug.

### IoU (Intersection over Union)

$$\text{IoU} = \frac{|A\cap B|}{|A\cup B|} = \frac{\text{area of overlap}}{\text{area of union}} \in [0,1]$$

The fundamental measure of box agreement. IoU ≥ 0.5 is the classic "correct detection"
threshold.

**IoU-based losses** improve on plain L1 box regression: **GIoU** (handles non-overlapping
boxes), **DIoU** (adds centre distance), **CIoU** (adds aspect ratio).

### Non-Maximum Suppression (NMS)

Detectors emit many overlapping boxes for one object. NMS deduplicates:

```
sort boxes by confidence, descending
while boxes remain:
    keep the highest-confidence box b
    remove every remaining box with IoU(b, ·) > threshold   # typically 0.5
```

Per class. **Soft-NMS** decays confidence instead of deleting, which helps with genuinely
overlapping objects (a crowd of people).

### Architecture families

| Family | Approach | Speed | Accuracy | Examples |
|--------|----------|-------|----------|----------|
| **Two-stage** | Propose regions, then classify | Slower | Higher | R-CNN → Fast R-CNN → **Faster R-CNN** |
| **One-stage (anchor)** | Dense predictions over a grid of anchors | Fast | Good | YOLO v1–v8, SSD, RetinaNet |
| **One-stage (anchor-free)** | Predict centres/keypoints directly | Fast | Good | FCOS, CenterNet, YOLOX |
| **Transformer** | Set prediction, no NMS needed | Medium | High | DETR, Deformable DETR, DINO |

**Faster R-CNN** introduced the **Region Proposal Network** — a small CNN that proposes object
regions, making the whole pipeline end-to-end trainable and fast enough to be practical.

**YOLO ("You Only Look Once")** divides the image into a grid; each cell predicts boxes,
objectness, and class probabilities in a single forward pass. Real-time, and the practical
default for most applications.

**RetinaNet** introduced **focal loss** to fix the extreme foreground/background imbalance in
dense detectors (100,000 candidate locations, a handful of objects):

$$FL(p_t) = -\alpha_t(1-p_t)^\gamma\log(p_t)$$

**DETR** treats detection as direct set prediction: a transformer outputs a fixed set of
predictions matched to ground truth by the **Hungarian algorithm** (optimal bipartite
matching). No anchors, no NMS — a genuinely cleaner formulation, though slower to converge.

### Feature Pyramid Network (FPN)
Objects appear at many scales. FPN combines high-resolution/low-semantic early features with
low-resolution/high-semantic deep features via a top-down pathway with lateral connections.
Nearly every modern detector uses one.

### Evaluation: mAP

**Average Precision (AP)** = area under the precision–recall curve for one class.
**mAP** = mean AP over classes.

| Metric | Meaning |
|--------|---------|
| `mAP@0.5` | IoU threshold 0.5 (Pascal VOC) |
| `mAP@0.5:0.95` | Averaged over IoU 0.5→0.95 in steps of 0.05 — **the COCO standard**, much stricter |
| `mAP_small/medium/large` | Broken down by object size — always check this; small objects are usually the weak spot |

---

## 5. Segmentation

Classify **every pixel**.

| Type | Output | Distinguishes instances? |
|------|--------|-------------------------|
| **Semantic** | Class label per pixel | No — all cars are "car" |
| **Instance** | Mask per object | Yes — car 1, car 2 |
| **Panoptic** | Semantic + instance combined | Yes, and covers background "stuff" too |

### Key architectures

**FCN (Fully Convolutional Network):** replaces the classifier's dense layers with
convolutions, so the network outputs a spatial map instead of a vector. Upsamples with
transposed convolutions.

**U-Net:** encoder–decoder with **skip connections** joining matching resolution levels. The
encoder captures context (what) while losing spatial precision; the skips restore fine detail
(where). Designed for biomedical images and still the default for medical segmentation and
the backbone of diffusion models.

```
    input ──conv──┐                          ┌──> output
                  ├─ 64 ────────skip────────►┤
                  ├─ 128 ───────skip────────►┤
                  ├─ 256 ───────skip────────►┤
                  └─ 512 (bottleneck) ───────┘
        (downsample)                (upsample)
```

**DeepLab:** atrous/dilated convolutions expand the receptive field without losing resolution;
**ASPP** (Atrous Spatial Pyramid Pooling) captures multiple scales in parallel.

**Mask R-CNN:** Faster R-CNN + a parallel mask branch, plus **RoIAlign** (bilinear sampling
instead of quantized RoIPool) which fixes the misalignment that destroys mask quality.

**SAM (Segment Anything):** a promptable foundation model — click a point or draw a box and it
segments the object, zero-shot, on essentially any image.

### Segmentation losses

**Dice loss** — directly optimizes overlap, handling class imbalance far better than pixel-wise
cross-entropy (where a mostly-background image makes "predict all background" a strong local
optimum):

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_i p_ig_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}$$

**IoU/Jaccard loss:** $1 - \frac{\sum p_ig_i}{\sum p_i + \sum g_i - \sum p_ig_i}$.

**Combined** (`CE + Dice`) is the standard practical choice: cross-entropy gives stable
gradients, Dice targets the metric you actually report. **Tversky loss** generalizes Dice with
tunable FP/FN weighting.

### Metrics
Pixel accuracy (misleading with imbalance), **mIoU** (mean Intersection over Union per class —
the standard), Dice/F1 coefficient, boundary F1 for edge quality.

---

## 6. Vision Transformers

### ViT (Dosovitskiy et al., 2020)

**The core idea:** treat an image as a sequence of patches and feed it to a standard
transformer encoder.

```
1. Split the image into fixed patches (e.g. 16×16)
   224×224 → 196 patches of 16×16×3
2. Flatten each patch (768 values) and linearly project to d_model
3. Prepend a learnable [CLS] token
4. Add positional embeddings
5. Standard transformer encoder blocks
6. Classify from the [CLS] token's final representation
```

Number of patches: $N = HW/P^2$. For 224×224 with $P=16$: $N=196$ tokens.

### ViT vs CNN

| | CNN | ViT |
|---|-----|-----|
| **Inductive bias** | Strong (locality, translation equivariance) | Weak — must learn spatial structure from data |
| **Data efficiency** | Good on small datasets | **Needs large data** (or heavy augmentation/distillation) |
| **Receptive field** | Grows with depth | Global from layer 1 |
| **Scaling** | Saturates | Keeps improving with data and size |
| **Compute** | $O(HW)$ | $O(N^2)$ in patches |

**The lesson:** with enough data, learned representations beat hand-designed inductive biases —
but below that threshold, the CNN's built-in assumptions are a genuine advantage. On a small
custom dataset, a fine-tuned ConvNeXt or ResNet is often the better choice.

**Swin Transformer** reintroduces hierarchy: attention within local windows that *shift*
between layers, giving linear complexity in image size and multi-scale features suitable for
detection and segmentation.

**DeiT** showed ViTs can train on ImageNet alone using distillation and strong augmentation.

---

## 7. Self-Supervised and Multimodal Vision

### Self-supervised pretraining

Learn representations without labels, then fine-tune on a small labeled set.

| Method | Idea |
|--------|------|
| **SimCLR** | Contrastive: two augmented views of the same image should embed together; different images apart. Needs large batches for enough negatives |
| **MoCo** | Contrastive with a momentum encoder and a queue of negatives — removes the large-batch requirement |
| **BYOL** | No negatives at all — a student predicts a momentum teacher's output |
| **DINO / DINOv2** | Self-distillation; attention maps segment objects *without ever being trained to* |
| **MAE** | Mask 75% of patches, reconstruct them. Simple, scalable, very effective |

**InfoNCE / NT-Xent contrastive loss:**

$$\mathcal{L} = -\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k\ne i}\exp(\text{sim}(z_i,z_k)/\tau)}$$

where $\text{sim}$ is cosine similarity and $\tau$ is a temperature (~0.07). The numerator is
the positive pair; the denominator includes all negatives.

### CLIP — connecting vision and language

Train an image encoder and a text encoder jointly on 400M image–caption pairs, with a
contrastive objective over the batch: matched (image, caption) pairs should have high cosine
similarity, all other pairs low.

**Why this mattered:** it produces a **shared embedding space** for images and text, enabling
**zero-shot classification** — to classify into arbitrary new categories, just embed the text
"a photo of a {class}" for each candidate and pick the nearest. No retraining, no labels.

CLIP embeddings underpin text-to-image generation (guiding diffusion), semantic image search,
and multimodal LLMs (which typically project image encoder outputs into the language model's
token space).

---

## 8. Generative Models

### GANs (Generative Adversarial Networks)

Two networks in a minimax game: a **generator** $G$ maps noise to images; a **discriminator**
$D$ tries to tell real from fake.

$$\min_G\max_D\; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

**Notorious failure modes:** mode collapse (the generator produces only a few outputs),
training instability, and no meaningful convergence metric. Mitigations: Wasserstein GAN with
gradient penalty, spectral normalization, progressive growing (StyleGAN).

**Where GANs still win:** speed. A single forward pass generates an image, versus dozens of
denoising steps for diffusion. Also super-resolution and image-to-image translation
(pix2pix, CycleGAN).

### Diffusion models — the current state of the art

**The idea:** learn to reverse a gradual noising process.

**Forward process** (fixed, no learning) — add Gaussian noise over $T$ steps:

$$q(\mathbf{x}_t\mid\mathbf{x}_{t-1}) = \mathcal{N}\big(\mathbf{x}_t;\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\;\beta_t\mathbf{I}\big)$$

A useful property: you can jump to any timestep in closed form. With
$\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$:

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon, \qquad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I})$$

**Reverse process** (learned) — a network $\boldsymbol\epsilon_\theta$ predicts the noise that
was added, and the training objective simplifies remarkably:

$$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\left[\big\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t)\big\|^2\right]$$

**That's just MSE on noise prediction.** Stable, well-behaved training — the opposite of GANs,
and the main reason diffusion took over.

**Sampling:** start from pure noise $\mathbf{x}_T\sim\mathcal{N}(0,I)$ and iteratively denoise
$T\to0$. DDPM needs ~1000 steps; **DDIM** and modern solvers (DPM-Solver++) get comparable
quality in 20–50.

**Latent diffusion (Stable Diffusion):** run the diffusion process in a VAE's compressed latent
space (e.g. 64×64×4) instead of pixel space (512×512×3) — roughly 48× less data to denoise.
This is what made high-resolution generation affordable.

**Classifier-free guidance** — how text prompts steer generation. Train with the conditioning
randomly dropped, then at sampling time extrapolate away from the unconditional prediction:

$$\tilde{\boldsymbol\epsilon} = \boldsymbol\epsilon_\theta(\mathbf{x}_t,\varnothing) + w\big(\boldsymbol\epsilon_\theta(\mathbf{x}_t, c) - \boldsymbol\epsilon_\theta(\mathbf{x}_t,\varnothing)\big)$$

The guidance scale $w$ (typically 7–12) trades diversity for prompt adherence. Too high and
images become oversaturated and rigid.

**Control mechanisms:** ControlNet (condition on edges, depth, or pose), inpainting/outpainting,
img2img, IP-Adapter (image prompts), LoRA (style/subject adaptation — the same technique as in
[Module 06 §6](06_nlp_and_llms.md#lora-low-rank-adaptation)).

### Comparing generative families

| | VAE | GAN | Diffusion | Autoregressive |
|---|-----|-----|-----------|----------------|
| Sample quality | Blurry | Sharp | **Best** | High |
| Training stability | Stable | **Unstable** | Stable | Stable |
| Sampling speed | Fast | **Fast** | Slow (iterative) | Slow |
| Mode coverage | Good | Poor (collapse) | **Excellent** | Excellent |
| Likelihood | ELBO bound | None | Bound | **Exact** |

### Evaluating generative models

| Metric | Measures | Caveat |
|--------|----------|--------|
| **FID** (Fréchet Inception Distance) | Distance between real and generated feature distributions — lower is better | The standard, but sensitive to sample count and implementation |
| **IS** (Inception Score) | Quality + diversity | Deprecated; ignores the real distribution |
| **CLIP score** | Image–text alignment | For text-to-image |
| **Human preference** | Actual quality | The real answer; expensive |

$$\text{FID} = \|\boldsymbol\mu_r-\boldsymbol\mu_g\|^2 + \text{Tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right)$$

---

## 9. Practical Vision Engineering

### The data reality
Vision projects are won on data, not architecture. Budget most of your time for collection,
labeling quality, and class balance. **Label noise is the dominant error source** in most
real-world vision datasets — audit your labels before blaming the model. Tools: CVAT, Label
Studio, Roboflow.

### Handling common problems

| Problem | Approach |
|---------|----------|
| Small dataset | Transfer learning + heavy augmentation + freeze most layers |
| Class imbalance | Weighted loss, focal loss, oversample rare classes, stratified batches |
| Small objects | Higher input resolution, FPN, tile large images into overlapping crops |
| Varying image sizes | Resize with letterboxing (preserve aspect ratio), or use a fully-convolutional model |
| Domain shift (lab → field) | Domain-specific augmentation, fine-tune on target data, test-time adaptation |
| Real-time requirement | MobileNet/YOLO-nano, quantization, TensorRT/ONNX, lower resolution |
| Video | Frame sampling + temporal smoothing; track instead of detecting every frame |

### Deployment
Export to **ONNX** for portability; **TensorRT** for NVIDIA GPUs; **CoreML** for iOS;
**TFLite** for Android/edge. Optimizations: INT8 quantization (~4× smaller, ~2–3× faster with
minimal accuracy loss), pruning, knowledge distillation into a small student model, batching,
and half-precision.

**Always benchmark end-to-end**, including image decode and preprocessing — these often
dominate the actual latency budget while everyone optimizes the model.

---

## Self-check

1. Compute the output shape of a 3×3 conv, stride 2, padding 1, on a 224×224×3 input with 64 filters. How many parameters?
2. Why does U-Net need skip connections?
3. Explain IoU and why mAP@0.5:0.95 is stricter than mAP@0.5.
4. Why does focal loss exist, and what problem in dense detection does it solve?
5. Why do ViTs need more data than CNNs?
6. Write the diffusion training objective and explain why it's so much more stable than a GAN's.
7. What does classifier-free guidance do, and what does raising $w$ trade away?
8. Why must segmentation masks be resized with nearest-neighbour interpolation?

---

**Next:** [08 — Reinforcement Learning →](08_reinforcement_learning.md)
