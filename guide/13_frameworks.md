# 13 — ML Frameworks: scikit-learn, TensorFlow/Keras, PyTorch

> The tools you actually build models with. Which framework for which job, the same problem
> solved in all three, and the API patterns you'll use every day.

📓 **Runnable companion notebook:
[`notebooks/13_framework_comparison.ipynb`](../notebooks/13_framework_comparison.ipynb)** —
solves one classification task in all three frameworks, plots the training curves, and ends
with a comparison table and exercises. Committed without outputs (keeps git diffs readable) —
run it yourself.

Also: `python/04_sklearn_machine_learning.py`, `python/05_tensorflow_keras_deep_learning.py`,
`python/06_pytorch_deep_learning.py`, `python/10_xgboost_lightgbm.py`.

```bash
pip install -r requirements.txt
pip install tensorflow torch          # optional; sections are skipped if absent
jupyter notebook notebooks/13_framework_comparison.ipynb
```

> **Heads-up:** TensorFlow and PyTorch each bundle their own OpenMP runtime, and on some
> platforms loading both into one kernel segfaults — typically at the first PyTorch operation
> after TensorFlow has trained. If that happens, restart the kernel and run *one* deep learning
> section per session; every section runs independently. It's one practical reason production
> images normally pin a single deep learning framework.

---

## Table of contents

1. [Choosing a framework](#1-choosing-a-framework)
2. [The same problem in all three](#2-the-same-problem-in-all-three)
3. [scikit-learn](#3-scikit-learn)
4. [TensorFlow and Keras](#4-tensorflow-and-keras)
5. [PyTorch](#5-pytorch)
6. [Gradient boosting libraries](#6-gradient-boosting-libraries)
7. [Hugging Face](#7-hugging-face)
8. [API Rosetta stone](#8-api-rosetta-stone)
9. [Use case → framework guide](#9-use-case--framework-guide)
10. [Common mistakes per framework](#10-common-mistakes-per-framework)

---

## 1. Choosing a framework

### The honest decision tree

```
Tabular data (rows and columns)?
├── < 1,000 rows            → scikit-learn (logistic regression, small RF)
├── 1k – 10M rows           → XGBoost / LightGBM / CatBoost   ← the correct default
└── Need a quick baseline   → scikit-learn, always

Images / text / audio / any deep learning?
├── Research, custom architectures, LLMs  → PyTorch      ← the field standard
├── Fast prototyping, clean high-level API → Keras 3
├── Mobile / edge / embedded deployment    → TensorFlow Lite, ONNX, ExecuTorch
└── Pretrained models of any kind          → Hugging Face (sits on PyTorch)

Classical unsupervised (clustering, PCA)?  → scikit-learn
Production serving of a small model?       → scikit-learn + FastAPI, or ONNX
```

### At a glance

| | **scikit-learn** | **TensorFlow/Keras** | **PyTorch** |
|---|---|---|---|
| **Best for** | Classical ML, tabular, preprocessing | Production DL, mobile/edge, clean API | Research, LLMs, custom architectures |
| **Learning curve** | Gentle | Gentle (Keras) / steep (raw TF) | Moderate |
| **Core abstraction** | Estimator (`fit`/`predict`) | Layer + Model (`compile`/`fit`) | `nn.Module` + explicit training loop |
| **Training loop** | Hidden | Hidden (`fit`) or custom | **You write it** |
| **Graph mode** | N/A | Static by default (`@tf.function`) | Dynamic (eager); compile with `torch.compile` |
| **Debugging** | Trivial | Harder in graph mode | Easy — it's just Python |
| **GPU** | ❌ (CPU only) | ✅ | ✅ |
| **Deep learning** | ⚠️ Toy `MLPClassifier` only | ✅ | ✅ |
| **Deployment** | pickle/ONNX | TF Serving, TFLite, TF.js | TorchServe, ONNX, ExecuTorch |
| **Research share** | — | Low | **Dominant** |
| **Ecosystem** | Enormous for classical ML | Large, enterprise | Enormous for DL/LLMs |

> **The honest summary for 2026:** learn **scikit-learn** (you'll use it in every project, even
> deep learning ones, for preprocessing and metrics) and **PyTorch** (the research and LLM
> standard). Learn Keras if you join a team using it or you want the fastest path from idea to
> trained network. Don't learn raw TensorFlow 1.x-style graph code — it's legacy.

> **Keras 3 is multi-backend.** It now runs on TensorFlow, **PyTorch**, or JAX via
> `KERAS_BACKEND=torch`. So "Keras vs PyTorch" is no longer strictly either/or — Keras is a
> high-level API you can put on top of PyTorch.

### Framework ≠ capability

All three can fit a neural network. The difference is **how much control you have and how much
you must write**:

```
scikit-learn   model.fit(X, y)                      ← you write 1 line, control almost nothing
Keras          model.compile(...); model.fit(...)   ← you write ~10 lines, control the architecture
PyTorch        for epoch: for batch: ...            ← you write ~20 lines, control everything
```

More control is only worth it when you need it. Reaching for PyTorch to fit a linear model on
500 rows is an error of judgment, not a display of skill.

---

## 2. The same problem in all three

**Task:** classify Iris flowers (4 features → 3 classes). Deliberately trivial, so the *API
differences* are the only thing on display.

📓 Run it: [`notebooks/13_framework_comparison.ipynb`](../notebooks/13_framework_comparison.ipynb)

### scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)),
])
model.fit(X_tr, y_tr)                                  # ← training is one line
print(accuracy_score(y_te, model.predict(X_te)))
```

### Keras

```python
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="softmax"),
])
model.compile(
    optimizer=keras.optimizers.Adam(1e-2),
    loss="sparse_categorical_crossentropy",     # sparse = integer labels, not one-hot
    metrics=["accuracy"],
)
model.fit(X_tr_scaled, y_tr, epochs=100, batch_size=16, verbose=0)
loss, acc = model.evaluate(X_te_scaled, y_te, verbose=0)
```

### PyTorch

```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

Xtr = torch.tensor(X_tr_scaled, dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.long)
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
criterion = nn.CrossEntropyLoss()          # expects LOGITS — no softmax in the model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):                    # ← you write the loop
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean().item()
```

### What the comparison shows

| | Lines to train | You control | You can get wrong |
|---|---|---|---|
| scikit-learn | 1 | Almost nothing | Almost nothing |
| Keras | 2 (`compile` + `fit`) | Architecture, optimizer, loss | Loss/activation mismatch |
| PyTorch | ~8 | Everything | `zero_grad`, `train`/`eval`, softmax double-application |

**Note the softmax asymmetry** — this trips up everyone moving between the two:
- **Keras**: `softmax` in the last layer + `sparse_categorical_crossentropy`.
- **PyTorch**: **no** softmax in the model; `nn.CrossEntropyLoss` applies log-softmax internally
  for numerical stability ([Module 01 §7.2](01_math_foundations.md#72-numerical-stability-tricks)).
  Adding softmax yourself applies it twice and quietly degrades training.

---

## 3. scikit-learn

**The most important library in this guide.** Even in a deep learning project you'll use it for
splitting, preprocessing, and metrics.

### The Estimator API — one interface for everything

Every object in scikit-learn follows the same contract, which is why the library is so easy to
learn:

| Method | Present on | Does |
|--------|-----------|------|
| `.fit(X, y)` | Everything | Learns parameters from data |
| `.predict(X)` | Predictors | Returns predictions |
| `.predict_proba(X)` | Classifiers | Returns class probabilities |
| `.transform(X)` | Transformers | Returns transformed data |
| `.fit_transform(X)` | Transformers | Fit then transform (training data only) |
| `.score(X, y)` | Predictors | Default metric (accuracy / R²) |

> **The single most important rule:** `fit_transform` on training data, `transform` **only** on
> validation/test data. Calling `fit_transform` on your test set leaks test statistics into
> preprocessing — see [Module 02 §5](02_ml_fundamentals.md#data-leakage--the-1-cause-of-great-model-terrible-production).

### Pipelines — the anti-leakage tool

A `Pipeline` chains transformers and a final estimator into one object. During cross-validation,
each transformer is re-fit **inside each fold**, which makes preprocessing leakage structurally
impossible.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

numeric = ["age", "income", "tenure"]
categorical = ["country", "plan"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),   # ← unseen categories at serve time
    ]), categorical),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", HistGradientBoostingClassifier(random_state=42)),
])

model.fit(X_train, y_train)     # one object: preprocessing + model
```

**Why this matters in production:** `joblib.dump(model, "model.joblib")` now saves the
*entire* pipeline. There's no way for serving code to scale features differently than training
did — the training/serving skew bug from
[Module 09 §10](09_mlops_and_deployment.md#useful-assertions) becomes impossible by construction.

`handle_unknown="ignore"` matters too: without it, a category unseen during training raises at
inference time and takes down your endpoint.

### Model selection

```python
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)

# Cross-validation — note the pipeline goes in, not the bare model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"AUC: {scores.mean():.3f} ± {scores.std():.3f}")     # always report the spread

# Hyperparameter search — "step__param" double-underscore syntax
param_dist = {
    "clf__max_depth": [3, 5, 7, None],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__max_iter": [100, 300, 500],
}
search = RandomizedSearchCV(
    model, param_dist, n_iter=20, cv=cv,
    scoring="roc_auc", random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
print(search.best_params_, search.best_score_)
```

Random search beats grid search at equal budget —
[Module 02 §10](02_ml_fundamentals.md#10-hyperparameter-tuning).

### Custom transformers

When you need domain-specific feature engineering inside a pipeline:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class RatioFeatures(BaseEstimator, TransformerMixin):
    """Adds debt-to-income ratio. Stateless, but must follow the API."""
    def fit(self, X, y=None):
        return self                       # nothing to learn
    def transform(self, X):
        X = X.copy()
        X["debt_to_income"] = X["debt"] / X["income"].replace(0, np.nan)
        return X
```

Inheriting `BaseEstimator` and `TransformerMixin` gives you `fit_transform`, `get_params`, and
`set_params` for free — which is what makes it work inside `GridSearchCV`.

### What scikit-learn is NOT for

- **GPU training** — it's CPU-only by design.
- **Serious deep learning** — `MLPClassifier` exists but is a toy: no GPU, no custom
  architectures, no convolutions, no attention.
- **Very large data** — most estimators need the data in memory. Use `partial_fit` (SGD-based
  estimators), or move to LightGBM/Spark.

### Use cases where scikit-learn is the right answer

| Use case | Approach |
|----------|----------|
| **Customer churn prediction** | `ColumnTransformer` + `HistGradientBoostingClassifier`, tune threshold for business cost |
| **Credit risk scoring** | `LogisticRegression` — regulators require interpretability; report coefficients as odds ratios |
| **Customer segmentation** | `StandardScaler` + `KMeans`, profile clusters |
| **Fraud detection (small scale)** | `IsolationForest`, or a classifier with `class_weight="balanced"` |
| **Sales forecasting (tabular)** | Lag features + `HistGradientBoostingRegressor` with `TimeSeriesSplit` |
| **Text classification baseline** | `TfidfVectorizer` + `LinearSVC` — still beats fine-tuned BERT at 1/1000 the cost on easy tasks |
| **Any baseline, ever** | `DummyClassifier` first, then logistic regression |

---

## 4. TensorFlow and Keras

**TensorFlow** is the numerical/graph engine; **Keras** is the high-level API you actually write.
Since Keras 3, it also runs on PyTorch and JAX backends.

### The three ways to build a model

**Sequential** — a linear stack of layers. Simple, limited.

```python
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax"),
])
```

**Functional** — a graph of layers. Handles multiple inputs/outputs, branches, skip connections.
**This is what you'll use for real models.**

```python
inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
skip = x                                        # ← branch
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.Add()([x, skip])                     # ← residual connection
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)
```

**Subclassing** — full control, PyTorch-style. Use when the forward pass has real logic.

```python
class MyModel(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.drop = layers.Dropout(0.3)
        self.out = layers.Dense(n_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.drop(x, training=training)     # ← dropout only active in training
        return self.out(x)
```

### compile → fit → evaluate

```python
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint("best.keras", save_best_only=True),
    ],
)
```

**Callbacks are Keras's best feature.** Early stopping with `restore_best_weights=True`,
LR scheduling, and checkpointing come free — in PyTorch you write all of that yourself (or
pull in Lightning).

### Choosing loss and final activation

| Task | Final layer | Loss | Label format |
|------|------------|------|--------------|
| Binary | `Dense(1, activation="sigmoid")` | `binary_crossentropy` | 0/1 |
| Multiclass | `Dense(K, activation="softmax")` | `sparse_categorical_crossentropy` | integers |
| Multiclass (one-hot) | `Dense(K, activation="softmax")` | `categorical_crossentropy` | one-hot |
| Multilabel | `Dense(K, activation="sigmoid")` | `binary_crossentropy` | multi-hot |
| Regression | `Dense(1)` (no activation) | `mse` / `huber` | float |

**Getting this pairing wrong is the #1 Keras bug.** Using `categorical_crossentropy` with
integer labels silently produces garbage rather than erroring clearly.

> **The safer alternative:** omit the final activation and pass `from_logits=True` to the loss.
> This is numerically more stable, and mirrors PyTorch's convention.

### tf.data — efficient input pipelines

```python
ds = (tf.data.Dataset.from_tensor_slices((X, y))
      .shuffle(buffer_size=10_000)
      .batch(32)
      .prefetch(tf.data.AUTOTUNE))       # ← overlaps data prep with training
```

`prefetch` and `cache` matter: on real datasets, an unoptimized input pipeline leaves the GPU
idle waiting for data, and you'll blame the model.

### Deployment — TensorFlow's genuine advantage

| Target | Tool |
|--------|------|
| Server | **TensorFlow Serving** (mature, high-performance gRPC/REST) |
| Mobile/embedded | **TFLite** — quantization, small binaries |
| Browser | **TensorFlow.js** |
| Cross-framework | ONNX export |

If your model must run on an Android phone or a microcontroller, TFLite is the most mature path.

### Use cases where Keras is a strong choice

| Use case | Approach |
|----------|----------|
| **Image classification** | Pretrained backbone from `keras.applications`, freeze, fine-tune |
| **Mobile image model** | MobileNetV3 → TFLite with INT8 quantization |
| **Tabular deep learning** | Functional API with embedding layers for categoricals |
| **Time-series forecasting** | LSTM/GRU or 1-D CNN over windowed sequences |
| **Fast prototyping** | Sequential + `fit` + `EarlyStopping` — idea to trained model in minutes |

---

## 5. PyTorch

**The research standard, and what essentially every LLM is written in.**

### Core building blocks

| Object | Role |
|--------|------|
| `torch.Tensor` | N-D array with autograd and GPU support |
| `nn.Module` | Base class for models and layers |
| `Dataset` / `DataLoader` | Data access and batching |
| `torch.optim` | Optimizers |
| `loss.backward()` | Populates `.grad` on every parameter |
| `optimizer.step()` | Applies the update |

### Defining a model

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_dim, hidden, n_classes, p_drop=0.3):
        super().__init__()                       # ← required, always
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)                       # ← LOGITS, no softmax
```

### Datasets and DataLoaders

```python
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    TabularDataset(X_train, y_train),
    batch_size=64, shuffle=True,      # shuffle=True for training
    num_workers=4, pin_memory=True,   # faster host→GPU transfer
)
val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=256, shuffle=False)
```

Three required methods: `__init__`, `__len__`, `__getitem__`. That's the whole interface.

### The training loop — annotated

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net(in_dim=20, hidden=64, n_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_val, patience, bad_epochs = float("inf"), 5, 0

for epoch in range(epochs):
    # ---- TRAIN ----
    model.train()                                  # enables dropout, BN batch stats
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()                      # 1. clear stale gradients
        logits = model(xb)                         # 2. forward
        loss = criterion(logits, yb)               # 3. loss
        loss.backward()                            # 4. backward
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 5. clip
        optimizer.step()                           # 6. update
    scheduler.step()

    # ---- VALIDATE ----
    model.eval()                                   # disables dropout, uses BN running stats
    val_loss = 0.0
    with torch.no_grad():                          # no graph → less memory, faster
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * len(yb)
    val_loss /= len(val_loader.dataset)

    # ---- EARLY STOPPING (you write this yourself) ----
    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break

model.load_state_dict(torch.load("best.pt"))
```

**The six-step inner loop is the thing to memorize.** Every PyTorch training script is this
loop with more decoration.

### The three lines everyone forgets

| Missing | Symptom |
|---------|---------|
| `optimizer.zero_grad()` | Gradients accumulate across batches; the model barely learns |
| `model.train()` / `model.eval()` | Dropout and BatchNorm behave wrongly; inference is worse and nondeterministic |
| `torch.no_grad()` at eval | Memory blows up; slower; possible OOM |

### Saving and loading

```python
# ✅ Recommended: save the state dict (weights only)
torch.save(model.state_dict(), "model.pt")
model = Net(in_dim=20, hidden=64, n_classes=3)         # rebuild architecture first
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()                                            # ← before inference

# ❌ torch.save(model, ...) pickles the class — fragile and unsafe across refactors
```

Use `weights_only=True` — loading an untrusted pickle executes arbitrary code
([Module 09 §5](09_mlops_and_deployment.md#serialization-formats)).

### Speed features worth knowing

| Feature | Gain |
|---------|------|
| `torch.compile(model)` | 1.3–2× with one line |
| AMP (`torch.autocast` + `GradScaler`) | ~2× speed, half the memory |
| `DistributedDataParallel` | Multi-GPU scaling |
| `pin_memory=True`, `num_workers>0` | Removes data-loading bottlenecks |

### Higher-level wrappers

Writing the loop is educational the first ten times; after that it's boilerplate you can get
wrong. **PyTorch Lightning** and **Hugging Face Accelerate** handle the loop, distributed
training, mixed precision, and checkpointing while keeping your model pure PyTorch.

### Use cases where PyTorch is the right answer

| Use case | Approach |
|----------|----------|
| **Fine-tuning an LLM** | Hugging Face `transformers` + `peft` (LoRA) — PyTorch underneath |
| **Custom architecture / research** | Subclass `nn.Module`; dynamic control flow just works |
| **Computer vision** | `torchvision` models + custom heads |
| **Diffusion / generative models** | `diffusers` |
| **Anything from a recent paper** | The reference implementation will be PyTorch |
| **RL** | Stable-Baselines3, CleanRL — all PyTorch |

---

## 6. Gradient boosting libraries

📄 `python/10_xgboost_lightgbm.py` · Theory:
[Module 03 §8](03_supervised_learning.md#8-boosting)

**For tabular data, these beat neural networks** at realistic dataset sizes. This is where most
real business value in ML actually lives.

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=2000,          # set high; early stopping decides the real number
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
)
```

| Library | Pick it when |
|---------|-------------|
| **XGBoost** | Battle-tested, best documentation, GPU support |
| **LightGBM** | Large datasets, need speed — usually the fastest |
| **CatBoost** | Many categorical features; the best out-of-the-box defaults |
| **sklearn `HistGradientBoosting`** | No extra dependency, sklearn-native, very solid |

All expose a scikit-learn-compatible API, so they drop straight into a `Pipeline` and
`RandomizedSearchCV`.

---

## 7. Hugging Face

The de-facto hub for pretrained models. Built on PyTorch.

```python
from transformers import pipeline

# Zero setup — downloads a model and runs it
clf = pipeline("sentiment-analysis")
print(clf("This guide is genuinely useful."))
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

```python
# Fine-tuning a text classifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out", num_train_epochs=3,
        per_device_train_batch_size=16, learning_rate=2e-5,   # small lr for fine-tuning
        eval_strategy="epoch", load_best_model_at_end=True,
    ),
    train_dataset=train_ds.map(tokenize, batched=True),
    eval_dataset=val_ds.map(tokenize, batched=True),
)
trainer.train()
```

| Package | Purpose |
|---------|---------|
| `transformers` | Pretrained models (text, vision, audio, multimodal) |
| `datasets` | Efficient dataset loading with memory mapping |
| `tokenizers` | Fast BPE/WordPiece implementations |
| `peft` | LoRA/QLoRA and other parameter-efficient fine-tuning |
| `accelerate` | Device placement and distributed training |
| `diffusers` | Diffusion models |
| `trl` | SFT, DPO, PPO for alignment |
| `sentence-transformers` | Embeddings for RAG and semantic search |

---

## 8. API Rosetta stone

The same concept across frameworks — useful when translating code you find online.

| Concept | scikit-learn | Keras | PyTorch |
|---------|-------------|-------|---------|
| **Define model** | `RandomForestClassifier()` | `keras.Sequential([...])` | `class Net(nn.Module)` |
| **Dense layer** | — | `layers.Dense(64)` | `nn.Linear(in, 64)` |
| **Conv layer** | — | `layers.Conv2D(32, 3)` | `nn.Conv2d(in_c, 32, 3)` |
| **Dropout** | — | `layers.Dropout(0.5)` | `nn.Dropout(0.5)` |
| **Batch norm** | — | `layers.BatchNormalization()` | `nn.BatchNorm1d/2d(n)` |
| **Train** | `.fit(X, y)` | `.fit(X, y, epochs=n)` | explicit loop |
| **Predict** | `.predict(X)` | `.predict(X)` | `model(x)` in `no_grad()` |
| **Probabilities** | `.predict_proba(X)` | `.predict(X)` (if softmax) | `softmax(logits, dim=1)` |
| **Loss (multiclass)** | internal | `"sparse_categorical_crossentropy"` | `nn.CrossEntropyLoss()` |
| **Optimizer** | internal (`solver=`) | `keras.optimizers.AdamW()` | `torch.optim.AdamW(params)` |
| **Train/eval mode** | N/A | automatic in `fit`/`predict` | `.train()` / `.eval()` |
| **Save** | `joblib.dump(m, f)` | `model.save("m.keras")` | `torch.save(m.state_dict(), f)` |
| **Load** | `joblib.load(f)` | `keras.models.load_model(f)` | `m.load_state_dict(torch.load(f))` |
| **GPU** | ❌ | automatic | `.to("cuda")` explicitly |
| **Early stopping** | `n_iter_no_change` | `EarlyStopping` callback | you write it |
| **Data batching** | N/A | `tf.data` or arrays | `Dataset` + `DataLoader` |

### Tensor operation equivalents

| Operation | NumPy | PyTorch | TensorFlow |
|-----------|-------|---------|-----------|
| Create | `np.array(x)` | `torch.tensor(x)` | `tf.constant(x)` |
| Zeros | `np.zeros((2,3))` | `torch.zeros(2,3)` | `tf.zeros((2,3))` |
| Matmul | `a @ b` | `a @ b` | `tf.matmul(a,b)` |
| Reshape | `a.reshape(2,3)` | `a.view(2,3)` / `a.reshape(2,3)` | `tf.reshape(a,(2,3))` |
| Transpose | `a.T` | `a.T` / `a.transpose(0,1)` | `tf.transpose(a)` |
| Concat | `np.concatenate([a,b])` | `torch.cat([a,b])` | `tf.concat([a,b],0)` |
| Add axis | `a[:, None]` | `a.unsqueeze(1)` | `tf.expand_dims(a,1)` |
| Sum axis | `a.sum(axis=1)` | `a.sum(dim=1)` | `tf.reduce_sum(a,1)` |
| Argmax | `a.argmax(axis=1)` | `a.argmax(dim=1)` | `tf.argmax(a,1)` |
| To NumPy | — | `t.detach().cpu().numpy()` | `t.numpy()` |

> `view` requires contiguous memory and fails after some transposes; `reshape` copies if needed.
> When in doubt, use `reshape` — or `.contiguous().view(...)`.

---

## 9. Use case → framework guide

| Your problem | Framework | Specifics |
|--------------|-----------|-----------|
| Predict churn from a database table | **scikit-learn / LightGBM** | `ColumnTransformer` + LightGBM, tune the threshold |
| Credit scoring with regulatory scrutiny | **scikit-learn** | Logistic regression; report odds ratios; SHAP for adverse-action notices |
| Classify product photos | **PyTorch** or **Keras** | Fine-tune a pretrained backbone |
| Detect defects on a production line | **PyTorch** | YOLO fine-tune; export to TensorRT for latency |
| Run a model on an Android phone | **TF/Keras** | TFLite + INT8 quantization |
| Chatbot over internal documents | **Hugging Face** | Embeddings + vector DB + LLM ([Module 06 §8](06_nlp_and_llms.md#8-retrieval-augmented-generation-rag)) |
| Fine-tune an LLM for a domain | **HF `transformers` + `peft`** | QLoRA on a quantized base model |
| Forecast demand per SKU | **LightGBM** | Lag features + `TimeSeriesSplit`; beat the seasonal-naive baseline |
| Segment customers | **scikit-learn** | `StandardScaler` + KMeans/HDBSCAN, then profile |
| Detect anomalies in server logs | **scikit-learn** | `IsolationForest`; autoencoder if high-dimensional |
| Train a game-playing agent | **PyTorch** | Stable-Baselines3 (PPO) |
| Reproduce a 2024 paper | **PyTorch** | The reference code will be PyTorch |
| Generate images | **PyTorch** | `diffusers` |
| Serve a small tabular model at scale | **scikit-learn** | Pipeline → joblib → FastAPI ([Module 09](09_mlops_and_deployment.md)) |

---

## 10. Common mistakes per framework

### scikit-learn

| Mistake | Consequence | Fix |
|---------|------------|-----|
| `fit_transform` on test data | Leakage; inflated scores | `transform` only |
| Scaling outside the pipeline before CV | Leakage across folds | Put the scaler *in* the `Pipeline` |
| Forgetting `stratify=y` on imbalanced splits | Folds with missing classes | `train_test_split(..., stratify=y)` |
| Thinking `C` is regularization *strength* | Backwards tuning | `C = 1/λ` — smaller `C` = **more** regularization |
| Using `.score()` on imbalanced data | Accuracy misleads | Choose an explicit `scoring=` |
| `OneHotEncoder` without `handle_unknown` | Crashes at serve time on new categories | `handle_unknown="ignore"` |

### Keras

| Mistake | Consequence | Fix |
|---------|------------|-----|
| `categorical_crossentropy` with integer labels | Silently wrong training | Use `sparse_categorical_crossentropy` |
| Softmax + `from_logits=True` | Double activation | Pick one convention |
| No `Input` shape on the first layer | Model not built; unclear errors | Add `layers.Input(shape=...)` |
| Forgetting `validation_data` | No early stopping signal | Always pass it |
| Not using `prefetch`/`cache` | GPU idles waiting on data | `tf.data` with `AUTOTUNE` |

### PyTorch

| Mistake | Consequence | Fix |
|---------|------------|-----|
| Missing `optimizer.zero_grad()` | Gradients accumulate; model won't learn | Call it every step |
| Missing `model.eval()` | Dropout/BN active at inference | Toggle modes explicitly |
| Softmax in the model **and** `CrossEntropyLoss` | Double softmax; degraded training | Output raw logits |
| Missing `torch.no_grad()` in eval | Memory blowup, OOM | Wrap evaluation |
| `loss` instead of `loss.item()` when logging | Retains the graph → memory leak | Use `.item()` |
| Wrong label dtype | Cryptic error | Labels must be `torch.long` for `CrossEntropyLoss` |
| Model on GPU, data on CPU | `RuntimeError: Expected all tensors on same device` | `.to(device)` both |

---

## Self-check

1. Why does putting a scaler inside a `Pipeline` prevent leakage during cross-validation?
2. Why does PyTorch's `nn.CrossEntropyLoss` expect logits rather than probabilities?
3. What breaks if you forget `model.eval()` before inference? Name two specific layers.
4. In scikit-learn's `LogisticRegression`, does a larger `C` mean more or less regularization?
5. Which framework would you pick for 50,000 rows of tabular data, and why not a neural network?
6. What do `model.train()` and `optimizer.zero_grad()` each do, and what happens without them?
7. Why is `torch.save(model.state_dict())` preferred over `torch.save(model)`?
8. You have integer labels and used `categorical_crossentropy` in Keras. What happens?

---

**Back to:** [Guide index](README.md) · **Related:** [05 — Deep Learning](05_deep_learning.md) · [09 — MLOps](09_mlops_and_deployment.md)
