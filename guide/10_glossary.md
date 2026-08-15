# 10 — Glossary (A–Z)

> Every term used in this guide, defined precisely, with a pointer to where it's developed in
> depth. Written to be turned into flashcards.

**[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [X](#x) · [Z](#z)**

---

## A

**A/B test** — Randomized controlled experiment comparing two variants on live traffic to
measure causal impact on a business metric. → [09 §6](09_mlops_and_deployment.md#6-deployment-strategies)

**Ablation study** — Systematically removing components of a system to measure each one's
contribution. The standard way to justify design choices in research.

**Accuracy** — Fraction of correct predictions, $(TP+TN)/(TP+TN+FP+FN)$. Misleading on
imbalanced data. → [02 §7](02_ml_fundamentals.md#7-evaluation-metrics)

**Activation function** — The non-linearity applied to a neuron's weighted sum. Without it,
stacked layers collapse into a single linear map. → [05 §3](05_deep_learning.md#3-activation-functions)

**Actor–critic** — RL architecture with a policy network (actor) and a value network (critic)
that supplies low-variance advantage estimates. → [08 §8](08_reinforcement_learning.md#8-policy-gradient-methods)

**AdaBoost** — Boosting algorithm that reweights misclassified examples so subsequent weak
learners focus on them. → [03 §8](03_supervised_learning.md#adaboost)

**AdaGrad** — Optimizer with per-parameter learning rates scaled by accumulated squared
gradients. Learning rate decays to zero over time. → [01 §5.4](01_math_foundations.md#54-momentum-and-adaptive-methods)

**Adam** — Adaptive Moment Estimation. Combines momentum (first moment) and RMSProp (second
moment) with bias correction. The default deep learning optimizer. → [01 §5.4](01_math_foundations.md#54-momentum-and-adaptive-methods)

**AdamW** — Adam with *decoupled* weight decay applied directly to weights rather than through
the adaptive scaling. Use this instead of Adam.

**Adversarial example** — Input with a small, often imperceptible perturbation that causes
misclassification.

**Advantage function** — $A(s,a) = Q(s,a) - V(s)$; how much better an action is than the
state's average. → [08 §3](08_reinforcement_learning.md#3-value-functions-and-bellman-equations)

**Agent** — (RL) The decision-making learner. (LLM) A system that loops between reasoning and
tool use to accomplish a task. → [06 §9](06_nlp_and_llms.md#9-tool-use-and-agents)

**AI winter** — Historical periods of collapsed funding and interest in AI, following
overpromising. Notably after the perceptron's XOR limitation was publicized (1969).

**Alignment** — Making a model's behaviour match human intentions and values. → [06 §7](06_nlp_and_llms.md#7-alignment)

**ANN (Approximate Nearest Neighbour)** — Sublinear-time similarity search that trades exactness
for speed. HNSW, IVF, PQ. The core of vector databases. → [06 §8](06_nlp_and_llms.md#retrieval)

**Anomaly detection** — Identifying data points that deviate from the expected pattern.
→ [04 §9](04_unsupervised_learning.md#9-anomaly-detection)

**ARIMA** — AutoRegressive Integrated Moving Average; classical time-series forecasting model.
→ [03 §12](03_supervised_learning.md#12-time-series)

**Attention** — Mechanism computing a weighted average of values, where weights come from
query–key similarity. The core of transformers. → [05 §9](05_deep_learning.md#9-attention-and-transformers)

**Autoencoder** — Network trained to reconstruct its input through a bottleneck, learning a
compressed representation. → [04 §8](04_unsupervised_learning.md#8-autoencoders)

**Autodiff (automatic differentiation)** — Mechanically computing exact derivatives by applying
the chain rule over a computational graph. → [05 §4](05_deep_learning.md#automatic-differentiation)

**Autoregressive** — Generating a sequence one element at a time, each conditioned on all
previous ones. How GPT-style models work. → [06 §4](06_nlp_and_llms.md#4-language-modelling)

**AUC (Area Under the Curve)** — Usually ROC-AUC: the probability a random positive is scored
above a random negative. → [02 §7](02_ml_fundamentals.md#threshold-free-metrics)

---

## B

**Backpropagation** — Efficient computation of loss gradients w.r.t. all parameters by applying
the chain rule backwards through the network with dynamic programming. → [05 §4](05_deep_learning.md#4-backpropagation)

**Bagging (Bootstrap Aggregating)** — Training models on bootstrap samples and averaging.
Reduces variance. → [03 §7](03_supervised_learning.md#7-bagging-and-random-forests)

**Baseline** — A simple reference model (majority class, mean, existing rule) that any real
model must beat to justify its complexity.

**Batch** — A group of examples processed together in one forward/backward pass.

**Batch normalization** — Normalizing activations across the batch dimension, with learned
scale and shift. → [05 §6.2](05_deep_learning.md#62-batch-normalization)

**Bayes' theorem** — $P(H|E) = P(E|H)P(H)/P(E)$. → [01 §3.4](01_math_foundations.md#34-bayes-theorem)

**Bellman equation** — Recursive relation expressing a state's value as the immediate reward
plus the discounted value of the successor state. → [08 §3](08_reinforcement_learning.md#3-value-functions-and-bellman-equations)

**BERT** — Bidirectional Encoder Representations from Transformers; encoder-only model trained
with masked language modelling. → [06 §5](06_nlp_and_llms.md#5-pretraining)

**Bias (statistical)** — Systematic error; the difference between an estimator's expected value
and the truth. → [02 §3](02_ml_fundamentals.md#3-the-biasvariance-tradeoff)

**Bias (neuron parameter)** — The learned additive constant $b$ in $\mathbf{w}^\top\mathbf{x}+b$.

**Bias–variance tradeoff** — Test error decomposes into bias², variance, and irreducible noise;
reducing one typically increases the other. → [02 §3](02_ml_fundamentals.md#3-the-biasvariance-tradeoff)

**BLEU** — n-gram precision metric for machine translation. → [06 §10](06_nlp_and_llms.md#10-evaluating-llm-systems)

**BM25** — Probabilistic keyword-ranking function; the backbone of classical search engines and
of hybrid RAG retrieval. → [06 §1](06_nlp_and_llms.md#1-classical-nlp-foundations)

**Boosting** — Sequentially training models where each corrects the ensemble's current errors.
Reduces bias. → [03 §8](03_supervised_learning.md#8-boosting)

**Bootstrap** — Resampling with replacement to estimate the sampling distribution of a
statistic. → [01 §4.6](01_math_foundations.md#46-bootstrap)

**BPE (Byte-Pair Encoding)** — Subword tokenization that iteratively merges the most frequent
symbol pairs. → [06 §2](06_nlp_and_llms.md#2-tokenization)

**Broadcasting** — NumPy/PyTorch rules for automatically expanding array shapes in element-wise
operations. → [01 §7.3](01_math_foundations.md#73-vectorization)

---

## C

**Calibration** — Whether predicted probabilities match observed frequencies. A model
predicting 0.7 should be right 70% of the time. → [02 §7](02_ml_fundamentals.md#calibration)

**CART** — Classification And Regression Trees; the standard greedy tree-growing algorithm.
→ [03 §6](03_supervised_learning.md#6-decision-trees)

**Catastrophic forgetting** — Loss of previously learned capability when fine-tuning on new
data. → [06 §6](06_nlp_and_llms.md#catastrophic-forgetting)

**Categorical distribution** — Distribution over $K$ discrete outcomes; the multiclass label
model behind softmax.

**Causal mask** — Attention mask preventing a position from attending to future positions,
making the model autoregressive. → [05 §9](05_deep_learning.md#masking)

**Central Limit Theorem** — The standardized sample mean converges to a normal distribution
regardless of the underlying distribution. → [01 §3.7](01_math_foundations.md#37-central-limit-theorem--law-of-large-numbers)

**Chain rule** — $\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)$. The mathematical basis of
backpropagation. → [01 §2.1](01_math_foundations.md#21-derivatives)

**Chain-of-thought (CoT)** — Prompting a model to produce intermediate reasoning steps, which
substantially improves multi-step performance. → [06 §6](06_nlp_and_llms.md#prompt-engineering)

**Chinchilla scaling** — Compute-optimal training uses ~20 tokens per parameter; earlier large
models were badly under-trained on data. → [06 §5](06_nlp_and_llms.md#scaling-laws)

**Classification** — Predicting a discrete class label.

**CLIP** — Contrastive Language-Image Pretraining; joint image–text embedding space enabling
zero-shot classification. → [07 §7](07_computer_vision.md#clip--connecting-vision-and-language)

**Clustering** — Grouping similar examples without labels. → [04](04_unsupervised_learning.md)

**CNN (Convolutional Neural Network)** — Network using convolution for local, translation-
equivariant, parameter-shared feature extraction. → [05 §7](05_deep_learning.md#7-convolutional-neural-networks-cnns)

**Concept drift** — $P(y|x)$ changes over time; the relationship itself shifts. → [09 §8](09_mlops_and_deployment.md#8-drift-detection)

**Confusion matrix** — Table of TP/FP/TN/FN counts. → [02 §7](02_ml_fundamentals.md#the-confusion-matrix)

**Contrastive learning** — Learning representations by pulling similar pairs together and
pushing dissimilar pairs apart. → [07 §7](07_computer_vision.md#7-self-supervised-and-multimodal-vision)

**Convex function** — A function whose chord lies above its graph; every local minimum is
global. → [01 §2.5](01_math_foundations.md#25-convexity)

**Convolution** — Sliding a learned kernel over an input to produce a feature map. → [05 §7](05_deep_learning.md#the-convolution-operation)

**Correlation** — Normalized covariance in $[-1,1]$; measures *linear* dependence only.
→ [01 §3.5](01_math_foundations.md#35-expectation-variance-covariance)

**Covariance** — $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$; how two variables co-vary.

**Covariate shift** — $P(x)$ changes while $P(y|x)$ stays fixed. → [02 §9](02_ml_fundamentals.md#distribution-shift)

**Cross-entropy** — $-\sum_x p(x)\log q(x)$; the standard classification loss, equal to NLL
under a categorical model. → [01 §6.2](01_math_foundations.md#62-cross-entropy)

**Cross-validation** — Rotating held-out folds to estimate generalization performance.
→ [02 §5](02_ml_fundamentals.md#k-fold-cross-validation)

**Curse of dimensionality** — In high dimensions data becomes sparse and distances become
uninformative. → [02 §8](02_ml_fundamentals.md#feature-selection)

---

## D

**Data augmentation** — Creating additional training examples via label-preserving transforms.
→ [05 §10](05_deep_learning.md#data-augmentation)

**Data leakage** — Information available at training time that won't be at prediction time.
The #1 cause of models that work offline and fail in production. → [02 §5](02_ml_fundamentals.md#data-leakage--the-1-cause-of-great-model-terrible-production)

**DBSCAN** — Density-based clustering that finds arbitrarily-shaped clusters and labels
outliers. → [04 §3](04_unsupervised_learning.md#3-dbscan-and-density-clustering)

**Decision boundary** — The surface in feature space where a classifier's predicted class
changes.

**Decision tree** — Model that recursively partitions the feature space with axis-aligned
splits. → [03 §6](03_supervised_learning.md#6-decision-trees)

**Deep learning** — ML with neural networks of many layers that learn hierarchical
representations. → [05](05_deep_learning.md)

**Determinant** — Scalar giving a linear map's volume-scaling factor; zero means singular.
→ [01 §1.7](01_math_foundations.md#17-matrix-inverse-and-solving-linear-systems)

**DDPM / Diffusion model** — Generative model that learns to reverse a gradual noising process.
→ [07 §8](07_computer_vision.md#diffusion-models--the-current-state-of-the-art)

**Dice loss** — Overlap-based segmentation loss that handles class imbalance well.
→ [07 §5](07_computer_vision.md#segmentation-losses)

**Dimensionality reduction** — Projecting data into fewer dimensions while retaining structure.
→ [04](04_unsupervised_learning.md)

**Discount factor ($\gamma$)** — Weight on future rewards in RL; controls the effective
planning horizon. → [08 §2](08_reinforcement_learning.md#the-return)

**Discriminative model** — Models $P(y|x)$ directly (logistic regression, most neural nets), in
contrast to generative models.

**Distillation** — Training a small student model to match a large teacher's outputs.

**Distribution shift** — Any change between training and deployment distributions.
→ [02 §9](02_ml_fundamentals.md#distribution-shift)

**DPO (Direct Preference Optimization)** — Alignment via a supervised loss on preference pairs,
eliminating the reward model and RL loop. → [06 §7](06_nlp_and_llms.md#dpo-direct-preference-optimization)

**Dropout** — Randomly zeroing units during training as a regularizer. → [02 §6](02_ml_fundamentals.md#other-regularizers)

**Dying ReLU** — A neuron stuck outputting zero for all inputs, with permanently zero gradient.
→ [05 §3](05_deep_learning.md#the-dying-relu-problem)

---

## E

**Early stopping** — Halting training when validation performance stops improving.

**Eigenvalue / eigenvector** — $A\mathbf{v} = \lambda\mathbf{v}$: a direction unchanged (only
scaled) by a linear map. → [01 §1.8](01_math_foundations.md#18-eigenvalues-and-eigenvectors)

**ELBO (Evidence Lower BOund)** — The variational objective maximized by VAEs: reconstruction
minus KL to the prior. → [04 §8](04_unsupervised_learning.md#variational-autoencoder-vae)

**Embedding** — A dense vector representation where geometric proximity encodes semantic
similarity. → [06 §3](06_nlp_and_llms.md#3-word-embeddings)

**EM (Expectation–Maximization)** — Iterative algorithm alternating between inferring latent
variables and updating parameters. → [04 §4](04_unsupervised_learning.md#expectationmaximization-em)

**Empirical risk** — Average loss on the training sample; the quantity actually minimized.
→ [02 §1](02_ml_fundamentals.md#1-what-machine-learning-actually-is)

**Ensemble** — Combining multiple models' predictions.

**Entropy** — $-\sum p\log p$; average uncertainty in a distribution. → [01 §6.1](01_math_foundations.md#61-entropy)

**Epoch** — One complete pass over the training dataset.

**Epsilon-greedy** — Exploration strategy: act randomly with probability $\varepsilon$,
greedily otherwise. → [08 §6](08_reinforcement_learning.md#exploration-strategies)

**Expectation** — $\mathbb{E}[X] = \sum x\,p(x)$; the probability-weighted average.

**Experience replay** — Storing RL transitions in a buffer and training on random minibatches
to break temporal correlation. → [08 §7](08_reinforcement_learning.md#7-deep-q-networks)

**Explainability** — Making a model's decisions understandable to humans. → [09 §12](09_mlops_and_deployment.md#explainability)

**Exploding gradient** — Gradients growing exponentially through layers/timesteps, producing
NaNs. Fixed by clipping. → [05 §4](05_deep_learning.md#vanishing-and-exploding-gradients)

---

## F

**F1 score** — Harmonic mean of precision and recall, $2PR/(P+R)$. → [02 §7](02_ml_fundamentals.md#classification-metrics)

**Feature** — An input variable used for prediction.

**Feature engineering** — Constructing informative input variables from raw data.
→ [02 §8](02_ml_fundamentals.md#8-feature-engineering--preprocessing)

**Feature store** — Infrastructure providing consistent feature definitions for training and
serving, with point-in-time correctness. → [09 §3](09_mlops_and_deployment.md#feature-stores)

**FID (Fréchet Inception Distance)** — Distance between real and generated image feature
distributions; the standard generative-model metric. → [07 §8](07_computer_vision.md#evaluating-generative-models)

**Few-shot learning** — Adapting to a task from a handful of examples, typically in-context.

**Fine-tuning** — Continuing training a pretrained model on task-specific data.
→ [06 §6](06_nlp_and_llms.md#6-adapting-a-pretrained-model)

**FlashAttention** — Tiled, IO-aware exact attention that never materializes the $L\times L$
score matrix. → [05 §9](05_deep_learning.md#complexity-and-the-long-context-problem)

**Focal loss** — Loss that down-weights easy examples to handle extreme class imbalance.
→ [02 §4](02_ml_fundamentals.md#classification-losses)

**Forward pass** — Computing outputs from inputs through the network.

**FPN (Feature Pyramid Network)** — Multi-scale feature fusion for detecting objects of varying
size. → [07 §4](07_computer_vision.md#feature-pyramid-network-fpn)

---

## G

**GAN (Generative Adversarial Network)** — Generator and discriminator trained in a minimax
game. → [07 §8](07_computer_vision.md#gans-generative-adversarial-networks)

**GAE (Generalized Advantage Estimation)** — Exponentially-weighted average of n-step
advantages, trading bias against variance. → [08 §8](08_reinforcement_learning.md#variance-reduction)

**Gaussian distribution** — The normal distribution; the default noise model in ML.
→ [01 §3.6](01_math_foundations.md#36-key-distributions)

**GELU** — Gaussian Error Linear Unit, $z\Phi(z)$; the standard transformer activation.
→ [05 §3](05_deep_learning.md#3-activation-functions)

**Generalization** — Performance on data not seen during training. The entire point of ML.

**Generative model** — Models the joint $P(x,y)$ or $P(x)$, enabling sampling of new data.

**Gini impurity** — $1-\sum p_k^2$; the default split criterion in decision trees.
→ [03 §6](03_supervised_learning.md#impurity-measures-classification)

**GloVe** — Word embeddings from factorizing a global co-occurrence matrix.

**GMM (Gaussian Mixture Model)** — Density model as a weighted sum of Gaussians; soft
clustering. → [04 §4](04_unsupervised_learning.md#4-gaussian-mixture-models--em)

**Gradient** — Vector of partial derivatives; points in the direction of steepest ascent.
→ [01 §2.2](01_math_foundations.md#22-partial-derivatives-and-the-gradient)

**Gradient accumulation** — Summing gradients over several micro-batches before updating, to
simulate a large batch. → [05 §5](05_deep_learning.md#batch-size)

**Gradient boosting** — Sequentially fitting models to the negative gradient of the loss.
→ [03 §8](03_supervised_learning.md#gradient-boosting)

**Gradient checkpointing** — Recomputing activations during the backward pass to trade compute
for memory.

**Gradient clipping** — Rescaling gradients that exceed a norm threshold. → [01 §5.6](01_math_foundations.md#56-second-order-and-other-methods)

**Gradient descent** — $\theta \leftarrow \theta - \eta\nabla J$. → [01 §5.2](01_math_foundations.md#52-gradient-descent)

**Greedy decoding** — Always selecting the highest-probability next token.
→ [06 §4](06_nlp_and_llms.md#decoding-strategies)

**GQA (Grouped-Query Attention)** — Heads share key/value projections, shrinking the KV cache.

---

## H

**Hallucination** — An LLM generating fluent, confident, factually wrong content.
→ [06 §12](06_nlp_and_llms.md#12-limitations-and-risks)

**Hessian** — Matrix of second partial derivatives; describes curvature. → [01 §2.3](01_math_foundations.md#23-jacobian-and-hessian)

**Hidden layer** — Any layer between input and output.

**Hinge loss** — $\max(0, 1-y f(x))$; the SVM loss. → [03 §5](03_supervised_learning.md#soft-margin-real-data-isnt-separable)

**HNSW** — Hierarchical Navigable Small World graph; the leading ANN index.

**Holdout** — Data set aside and not used for training.

**Huber loss** — Quadratic near zero, linear far from it; robust regression loss.
→ [02 §4](02_ml_fundamentals.md#regression-losses)

**Hyperparameter** — A configuration value set before training (learning rate, tree depth), as
opposed to learned parameters. → [02 §10](02_ml_fundamentals.md#10-hyperparameter-tuning)

---

## I

**i.i.d.** — Independent and identically distributed; the assumption underpinning most of ML
theory, and the one most often violated. → [02 §1](02_ml_fundamentals.md#key-terminology)

**Imbalanced data** — Highly unequal class frequencies. → [02 §9](02_ml_fundamentals.md#class-imbalance)

**Imputation** — Filling in missing values. → [02 §8](02_ml_fundamentals.md#missing-values)

**In-context learning** — A model adapting its behaviour from examples in the prompt, without
weight updates.

**Inductive bias** — Assumptions built into a model that let it generalize beyond the training
data. CNNs assume locality; trees assume axis-aligned splits.

**Inference** — Applying a trained model to new inputs. (In statistics, it instead means
drawing conclusions about parameters — the overloading is unfortunate.)

**Information gain** — Entropy reduction from a split; equivalent to mutual information.
→ [03 §6](03_supervised_learning.md#impurity-measures-classification)

**Instance segmentation** — Per-pixel classification that distinguishes individual objects.
→ [07 §5](07_computer_vision.md#5-segmentation)

**IoU (Intersection over Union)** — Overlap ratio between two regions; the core detection and
segmentation metric. → [07 §4](07_computer_vision.md#iou-intersection-over-union)

**Isolation Forest** — Anomaly detection by random partitioning; anomalies isolate quickly.
→ [04 §9](04_unsupervised_learning.md#9-anomaly-detection)

---

## J

**Jacobian** — Matrix of all first-order partial derivatives of a vector-valued function.
→ [01 §2.3](01_math_foundations.md#23-jacobian-and-hessian)

**Jensen–Shannon divergence** — Symmetric, bounded variant of KL divergence.

---

## K

**KD-tree** — Space-partitioning structure that accelerates nearest-neighbour search in low
dimensions.

**Kernel (ML)** — A function computing inner products in an implicit high-dimensional feature
space. → [03 §5](03_supervised_learning.md#the-dual-and-the-kernel-trick)

**Kernel (CNN)** — The learned filter slid across the input. → [05 §7](05_deep_learning.md#the-convolution-operation)

**Kernel trick** — Using $K(x,x')$ to work in a high-dimensional space without ever computing
the mapping. → [03 §5](03_supervised_learning.md#the-dual-and-the-kernel-trick)

**KL divergence** — $\sum p\log(p/q)$; asymmetric measure of how one distribution differs from
another. → [01 §6.3](01_math_foundations.md#63-kl-divergence)

**K-Means** — Clustering by iteratively assigning points to the nearest centroid and updating
centroids. → [04 §1](04_unsupervised_learning.md#1-k-means-clustering)

**kNN** — Predicting from the $k$ nearest training examples. → [03 §3](03_supervised_learning.md#3-k-nearest-neighbours-knn)

**KV cache** — Cached keys and values from previous tokens, making autoregressive generation
linear rather than quadratic. → [06 §11](06_nlp_and_llms.md#kv-cache)

---

## L

**L1 / L2 regularization** — Penalties $\lambda\|w\|_1$ (sparsity) and $\lambda\|w\|_2^2$
(shrinkage). → [02 §6](02_ml_fundamentals.md#6-regularization)

**Label smoothing** — Softening one-hot targets to reduce overconfidence.

**Lagrange multiplier** — Technique for constrained optimization. → [01 §2.6](01_math_foundations.md#26-lagrange-multipliers-constrained-optimization)

**Lasso** — Linear regression with an L1 penalty; performs feature selection.
→ [03 §1](03_supervised_learning.md#regularized-variants)

**Latent variable** — An unobserved variable inferred from data (cluster assignment, VAE code).

**Layer normalization** — Normalizing across features within each example; the transformer
standard. → [05 §6.3](05_deep_learning.md#63-layer-normalization)

**Learning rate** — The optimizer's step size; the most important hyperparameter in deep
learning. → [01 §5.2](01_math_foundations.md#52-gradient-descent)

**Likelihood** — $P(\text{data}|\theta)$ viewed as a function of $\theta$.
→ [01 §3.8](01_math_foundations.md#38-maximum-likelihood-estimation-mle)

**Linear regression** — Fitting a linear function by least squares. → [03 §1](03_supervised_learning.md#1-linear-regression)

**Logistic regression** — Linear model with a sigmoid output for binary classification.
→ [03 §2](03_supervised_learning.md#2-logistic-regression)

**Logit** — The pre-activation output of the final layer; also $\log(p/(1-p))$.

**Log-sum-exp trick** — Subtracting the max before exponentiating to prevent overflow.
→ [01 §7.2](01_math_foundations.md#72-numerical-stability-tricks)

**LoRA (Low-Rank Adaptation)** — Fine-tuning by learning a low-rank update $\Delta W = BA$ to
frozen weights. → [06 §6](06_nlp_and_llms.md#lora-low-rank-adaptation)

**Loss function** — The quantity minimized during training; encodes what "wrong" means.
→ [02 §4](02_ml_fundamentals.md#4-loss-functions)

**LSTM** — Recurrent unit with gates and an additive cell state that avoids vanishing
gradients. → [05 §8](05_deep_learning.md#lstm-long-short-term-memory)

---

## M

**mAP (mean Average Precision)** — The standard object detection metric. → [07 §4](07_computer_vision.md#evaluation-map)

**MAP (Maximum A Posteriori)** — Point estimate maximizing the posterior; MLE plus a prior.
Equivalent to regularization. → [01 §3.9](01_math_foundations.md#39-map-and-bayesian-inference)

**Markov property** — The future depends only on the present state, not the history.
→ [08 §2](08_reinforcement_learning.md#2-markov-decision-processes)

**MDP (Markov Decision Process)** — The formal framework for RL: states, actions, transitions,
rewards, discount. → [08 §2](08_reinforcement_learning.md#2-markov-decision-processes)

**Masked language modelling** — Predicting hidden tokens from bidirectional context (BERT).

**Matrix factorization** — Decomposing a matrix into a product of smaller ones.
→ [04 §6](04_unsupervised_learning.md#6-other-matrix-factorizations)

**MCC (Matthews Correlation Coefficient)** — Balanced classification metric using all four
confusion-matrix cells. → [02 §7](02_ml_fundamentals.md#classification-metrics)

**Mini-batch** — A subset of the training data used for one gradient step.

**MLE (Maximum Likelihood Estimation)** — Choosing parameters that maximize the probability of
the observed data. → [01 §3.8](01_math_foundations.md#38-maximum-likelihood-estimation-mle)

**MLOps** — Practices for reliably deploying and maintaining ML in production. → [09](09_mlops_and_deployment.md)

**Momentum** — Accumulating a velocity of past gradients to accelerate and stabilize descent.
→ [01 §5.4](01_math_foundations.md#54-momentum-and-adaptive-methods)

**MSE (Mean Squared Error)** — $\frac1n\sum(y-\hat y)^2$; the standard regression loss,
equivalent to MLE under Gaussian noise.

**Multicollinearity** — Highly correlated features, making $X^\top X$ near-singular and
coefficients unstable. → [03 §1](03_supervised_learning.md#when-the-closed-form-fails)

**Multi-head attention** — Running several attention operations in parallel with different
projections. → [05 §9](05_deep_learning.md#multi-head-attention)

**Mutual information** — $I(X;Y)$; the reduction in uncertainty about $X$ from knowing $Y$.
Captures non-linear dependence. → [01 §6.4](01_math_foundations.md#64-mutual-information)

---

## N

**Naive Bayes** — Bayes' rule plus a conditional-independence assumption. → [03 §4](03_supervised_learning.md#4-naive-bayes)

**Negative sampling** — Approximating an expensive softmax by contrasting against a few random
negatives. → [06 §3](06_nlp_and_llms.md#word2vec-mikolov-et-al-2013)

**NER (Named Entity Recognition)** — Extracting entities (people, places, dates) from text.

**Neural network** — Composition of parameterized linear maps and non-linearities.
→ [05](05_deep_learning.md)

**NLL (Negative Log-Likelihood)** — $-\log P(\text{data}|\theta)$; the general form of most ML
losses.

**NMS (Non-Maximum Suppression)** — Removing duplicate overlapping detections.
→ [07 §4](07_computer_vision.md#non-maximum-suppression-nms)

**No Free Lunch theorem** — Averaged over all problems, all algorithms perform equally; there
is no universally best model. → [02 §1](02_ml_fundamentals.md#no-free-lunch-theorem)

**Normalization** — Rescaling data or activations to a standard range or distribution.

---

## O

**Objective function** — The function being optimized (loss + regularization).

**Off-policy** — Learning about a policy different from the one collecting data (Q-learning).
→ [08 §6](08_reinforcement_learning.md#6-q-learning)

**On-policy** — Learning about the policy currently generating data (SARSA, PPO).

**One-hot encoding** — Representing a category as a binary indicator vector.

**ONNX** — Open Neural Network Exchange; a portable model format. → [09 §5](09_mlops_and_deployment.md#serialization-formats)

**OOB (Out-of-Bag)** — The ~36.8% of data excluded by each bootstrap sample, usable as free
validation. → [03 §7](03_supervised_learning.md#out-of-bag-oob-estimation)

**Optimizer** — The algorithm updating parameters from gradients. → [01 §5](01_math_foundations.md#part-5--optimization)

**Overfitting** — Fitting the training data's noise; low train error, high test error.
→ [02 §3](02_ml_fundamentals.md#3-the-biasvariance-tradeoff)

---

## P

**PCA (Principal Component Analysis)** — Projecting onto the directions of maximum variance.
→ [04 §5](04_unsupervised_learning.md#5-principal-component-analysis-pca)

**PEFT (Parameter-Efficient Fine-Tuning)** — Adapting a model by training a small fraction of
parameters (LoRA, adapters, prefix tuning). → [06 §6](06_nlp_and_llms.md#6-adapting-a-pretrained-model)

**Perceptron** — The original single-layer neuron with a step activation. → [05 §1](05_deep_learning.md#the-perceptron-rosenblatt-1958)

**Perplexity** — $e^{H}$; the standard language-model metric, interpretable as an effective
branching factor. → [01 §6.5](01_math_foundations.md#65-perplexity)

**Policy** — A mapping from states to actions (or action distributions) in RL.

**Policy gradient** — Directly optimizing a parameterized policy via
$\nabla J = \mathbb{E}[\nabla\log\pi\cdot Q]$. → [08 §8](08_reinforcement_learning.md#8-policy-gradient-methods)

**Pooling** — Downsampling a feature map by taking a max or mean over local windows.

**Positional encoding** — Injecting sequence-order information into a permutation-equivariant
transformer. → [05 §9](05_deep_learning.md#positional-encoding)

**Posterior** — $P(\theta|\text{data})$; belief after observing evidence.

**PPO (Proximal Policy Optimization)** — Policy gradient with a clipped objective limiting
update size. → [08 §9](08_reinforcement_learning.md#9-ppo-proximal-policy-optimization)

**Precision** — $TP/(TP+FP)$; correctness of positive predictions.

**Pretraining** — Large-scale self-supervised training before task-specific adaptation.
→ [06 §5](06_nlp_and_llms.md#5-pretraining)

**Prior** — $P(\theta)$; belief before observing data. Equivalent to a regularizer.

**Prompt injection** — Attack where untrusted content the model reads is interpreted as
instructions. → [06 §9](06_nlp_and_llms.md#where-agents-actually-break)

**PSI (Population Stability Index)** — The industry-standard drift metric.
→ [09 §8](09_mlops_and_deployment.md#statistical-tests)

**p-value** — $P(\text{data this extreme}\mid H_0)$. Not the probability that $H_0$ is true.
→ [01 §4.4](01_math_foundations.md#44-hypothesis-testing)

---

## Q

**Q-function** — $Q(s,a)$; expected return from taking action $a$ in state $s$.

**Q-learning** — Off-policy TD control that learns $Q^*$ directly. → [08 §6](08_reinforcement_learning.md#6-q-learning)

**QLoRA** — LoRA on top of a 4-bit quantized frozen base model. → [06 §6](06_nlp_and_llms.md#lora-low-rank-adaptation)

**Quantization** — Representing weights/activations in fewer bits (INT8, INT4).
→ [06 §11](06_nlp_and_llms.md#quantization)

---

## R

**RAG (Retrieval-Augmented Generation)** — Retrieving relevant documents and providing them as
context to the model. → [06 §8](06_nlp_and_llms.md#8-retrieval-augmented-generation-rag)

**Random forest** — Bagged decision trees with per-split feature subsampling.
→ [03 §7](03_supervised_learning.md#7-bagging-and-random-forests)

**Rank** — The number of linearly independent columns of a matrix. → [01 §1.6](01_math_foundations.md#16-linear-independence-span-rank)

**Recall** — $TP/(TP+FN)$; fraction of actual positives found.

**Receptive field** — The region of input influencing a particular output unit.

**Regression** — Predicting a continuous value.

**Regularization** — Anything that reduces generalization error without reducing training error.
→ [02 §6](02_ml_fundamentals.md#6-regularization)

**ReLU** — $\max(0,z)$; the default hidden activation.

**Reparameterization trick** — Writing $z = \mu + \sigma\epsilon$ so gradients flow through a
sampling step. → [04 §8](04_unsupervised_learning.md#variational-autoencoder-vae)

**Residual connection** — $y = F(x) + x$; the skip path that makes very deep networks
trainable. → [05 §6.4](05_deep_learning.md#64-residual-connections)

**Reward hacking** — An agent maximizing the specified reward in unintended ways.
→ [08 §2](08_reinforcement_learning.md#reward-design--where-most-rl-projects-fail)

**Reward model** — A model trained on human preferences to score outputs, used in RLHF.

**Ridge regression** — Linear regression with an L2 penalty. → [03 §1](03_supervised_learning.md#regularized-variants)

**RLHF** — Reinforcement Learning from Human Feedback. → [06 §7](06_nlp_and_llms.md#rlhf-reinforcement-learning-from-human-feedback)

**RMSNorm** — LayerNorm without mean centering; used in modern LLMs.

**RNN** — Network with a recurrent hidden state for sequential data. → [05 §8](05_deep_learning.md#8-recurrent-neural-networks)

**RoPE (Rotary Position Embedding)** — Encoding position by rotating query/key vectors;
naturally relative and extends well. → [05 §9](05_deep_learning.md#positional-encoding)

**ROC curve** — TPR vs FPR across all thresholds. → [02 §7](02_ml_fundamentals.md#threshold-free-metrics)

**R²** — Fraction of target variance explained by the model.

---

## S

**SAC (Soft Actor-Critic)** — Maximum-entropy off-policy RL; strong for continuous control.

**Sampling (decoding)** — Drawing tokens from the model's output distribution rather than
taking the argmax. → [06 §4](06_nlp_and_llms.md#decoding-strategies)

**SARSA** — On-policy TD control using $(s,a,r,s',a')$. → [08 §5](08_reinforcement_learning.md#sarsa--on-policy-td-control)

**Scaling laws** — Power-law relationships between loss and compute/data/parameters.
→ [06 §5](06_nlp_and_llms.md#scaling-laws)

**Self-attention** — Attention where queries, keys, and values all come from the same sequence.

**Self-supervised learning** — Deriving labels from the data itself (next-token prediction,
masking). The engine of modern AI. → [02 §2](02_ml_fundamentals.md#by-supervision)

**Semantic segmentation** — Per-pixel class labels without distinguishing instances.

**SGD (Stochastic Gradient Descent)** — Gradient descent on mini-batches.
→ [01 §5.3](01_math_foundations.md#53-batch-stochastic-and-mini-batch)

**SHAP** — Shapley Additive exPlanations; game-theoretic feature attribution.
→ [09 §12](09_mlops_and_deployment.md#explainability)

**Sigmoid** — $1/(1+e^{-z})$; squashes to $(0,1)$ for binary probabilities.

**Silhouette score** — Clustering quality measure balancing cohesion and separation.
→ [04 §1](04_unsupervised_learning.md#choosing-k)

**Singular matrix** — Non-invertible; determinant zero; rank deficient.

**Softmax** — $e^{z_i}/\sum_j e^{z_j}$; converts logits to a probability distribution.

**SMOTE** — Synthetic Minority Over-sampling Technique; generates synthetic minority examples.
→ [02 §9](02_ml_fundamentals.md#class-imbalance)

**Sparse** — Mostly zeros.

**Speculative decoding** — A small draft model proposes tokens that the large model verifies in
one pass. → [06 §11](06_nlp_and_llms.md#throughput-techniques)

**Standardization** — $(x-\mu)/\sigma$; rescaling to zero mean and unit variance.

**Stationarity** — Statistical properties constant over time; required by classical time-series
models. → [03 §12](03_supervised_learning.md#12-time-series)

**Stride** — The step size when sliding a convolution kernel.

**Supervised learning** — Learning from labeled examples. → [03](03_supervised_learning.md)

**SVD (Singular Value Decomposition)** — $A = U\Sigma V^\top$; the universal matrix
factorization. → [01 §1.9](01_math_foundations.md#19-singular-value-decomposition-svd)

**SVM (Support Vector Machine)** — Maximum-margin classifier, kernelizable.
→ [03 §5](03_supervised_learning.md#5-support-vector-machines-svm)

**Support vector** — A training point on or inside the margin; the only points defining the SVM
boundary.

---

## T

**Target network** — A periodically-synced copy of the Q-network providing stable TD targets.
→ [08 §7](08_reinforcement_learning.md#7-deep-q-networks)

**TD (Temporal Difference) learning** — Updating value estimates toward a bootstrapped target.
→ [08 §5](08_reinforcement_learning.md#temporal-difference-td-learning)

**Temperature** — Softmax scaling controlling randomness in sampling.
→ [06 §4](06_nlp_and_llms.md#decoding-strategies)

**Tensor** — A multi-dimensional array.

**TF-IDF** — Term Frequency × Inverse Document Frequency; classical text weighting.

**Token** — The atomic unit a language model processes; typically a subword.
→ [06 §2](06_nlp_and_llms.md#2-tokenization)

**Trace** — Sum of a matrix's diagonal entries; equals the sum of its eigenvalues.

**Transfer learning** — Reusing a model trained on one task for another.
→ [05 §10](05_deep_learning.md#transfer-learning)

**Transformer** — Architecture built on self-attention; the basis of modern AI.
→ [05 §9](05_deep_learning.md#9-attention-and-transformers)

**t-SNE** — Non-linear visualization technique preserving local neighbourhoods.
→ [04 §7](04_unsupervised_learning.md#t-sne-t-distributed-stochastic-neighbour-embedding)

**TTFT (Time To First Token)** — Prefill latency; dominates perceived LLM responsiveness.

---

## U

**UMAP** — Faster alternative to t-SNE that better preserves global structure and can transform
new data. → [04 §7](04_unsupervised_learning.md#umap-uniform-manifold-approximation-and-projection)

**Underfitting** — Model too simple to capture the pattern; high train *and* test error.

**Universal Approximation Theorem** — A single hidden layer can approximate any continuous
function — but says nothing about how many units or whether you can find the weights.
→ [05 §2](05_deep_learning.md#universal-approximation-theorem)

**Unsupervised learning** — Learning structure from unlabeled data. → [04](04_unsupervised_learning.md)

**U-Net** — Encoder–decoder with skip connections; the segmentation and diffusion standard.
→ [07 §5](07_computer_vision.md#key-architectures)

---

## V

**VAE (Variational Autoencoder)** — Probabilistic autoencoder with a continuous, sampleable
latent space. → [04 §8](04_unsupervised_learning.md#variational-autoencoder-vae)

**Validation set** — Data used to tune hyperparameters and select models — not for the final
performance estimate. → [02 §5](02_ml_fundamentals.md#the-three-way-split)

**Value function** — $V(s)$; expected return from a state.

**Vanishing gradient** — Gradients shrinking exponentially through layers, stalling learning in
early layers. → [05 §4](05_deep_learning.md#vanishing-and-exploding-gradients)

**Variance (statistical)** — $\mathbb{E}[(X-\mu)^2]$; spread of a distribution.

**Variance (bias–variance)** — Error from sensitivity to the particular training sample.

**Vector database** — Storage and ANN search over embeddings. → [06 §8](06_nlp_and_llms.md#retrieval)

**ViT (Vision Transformer)** — Transformer applied to image patches. → [07 §6](07_computer_vision.md#6-vision-transformers)

---

## W

**Weight decay** — L2 regularization implemented as multiplicative shrinkage in the update.

**Weight sharing** — Reusing the same parameters across positions (CNNs) or timesteps (RNNs).

**Word2Vec** — Learning word embeddings by predicting context words.
→ [06 §3](06_nlp_and_llms.md#word2vec-mikolov-et-al-2013)

---

## X

**Xavier/Glorot initialization** — Weight init scaled by $2/(n_{in}+n_{out})$; for tanh/sigmoid.
→ [05 §6.1](05_deep_learning.md#61-weight-initialization)

**XGBoost** — Gradient boosting with second-order gradients and explicit regularization.
→ [03 §8](03_supervised_learning.md#xgboost--the-second-order-refinement)

**XOR problem** — The non-linearly-separable function a single perceptron cannot learn;
motivated hidden layers. → [05 §1](05_deep_learning.md#the-perceptron-rosenblatt-1958)

---

## Z

**Zero-shot learning** — Performing a task with no task-specific examples, relying on
pretrained knowledge.

**Z-score** — $(x-\mu)/\sigma$; how many standard deviations a value sits from the mean.

---

**Next:** [11 — Notation Reference →](11_notation.md)
