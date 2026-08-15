# 09 — MLOps and Deployment

> A model in a notebook creates zero value. This module covers everything between "it works on
> my machine" and "it reliably serves real users."

📄 Companion code: `python/11_joblib_pickle_serialization.py`, `python/12_fastapi_flask_serving.py`

---

## Table of contents

1. [Why ML systems are different](#1-why-ml-systems-are-different)
2. [The ML lifecycle](#2-the-ml-lifecycle)
3. [Data engineering for ML](#3-data-engineering-for-ml)
4. [Experiment tracking and reproducibility](#4-experiment-tracking-and-reproducibility)
5. [Model packaging and serving](#5-model-packaging-and-serving)
6. [Deployment strategies](#6-deployment-strategies)
7. [Monitoring](#7-monitoring)
8. [Drift detection](#8-drift-detection)
9. [Retraining](#9-retraining)
10. [Testing ML systems](#10-testing-ml-systems)
11. [Cost, scale, and reliability](#11-cost-scale-and-reliability)
12. [Responsible AI](#12-responsible-ai)

---

## 1. Why ML Systems Are Different

Traditional software: behaviour is fully specified by code. ML systems: behaviour emerges from
**code + data + model artifacts**, all three of which drift.

| Property | Traditional software | ML system |
|----------|---------------------|-----------|
| Correctness | Deterministic; testable with asserts | Statistical; "correct" is a distribution |
| Failure mode | Crashes loudly | **Degrades silently** |
| Versioning | Code | Code + data + model + features |
| Testing | Unit/integration tests | + data validation, model quality, behavioural tests |
| Decay | Stable until changed | **Decays continuously as the world changes** |
| Debugging | Read the stack trace | Trace through data, features, and training |

> **The defining risk: silent failure.** A broken API returns a 500 and pages someone. A broken
> model returns confident, plausible, wrong predictions for weeks. **Monitoring is not
> optional — it is the primary safety mechanism.**

### Technical debt in ML ("Hidden Technical Debt in ML Systems", Sculley et al.)

| Anti-pattern | Description |
|-------------|-------------|
| **Glue code** | 95% of the codebase is plumbing around a 5% modelling core |
| **Pipeline jungles** | Organically-grown, undocumented data transformations |
| **CACE principle** | "**C**hanging **A**nything **C**hanges **E**verything" — no feature is truly independent |
| **Feedback loops** | The model's predictions influence the data it later trains on |
| **Undeclared consumers** | Teams silently depending on your model output |
| **Configuration debt** | Hundreds of untested config knobs |

**Feedback loops deserve special attention.** A recommender trained on clicks shapes what users
see, which shapes future clicks. A fraud model that blocks transactions never learns whether
those transactions were actually fraud. Design explicit exploration (hold out a random slice
from the model's influence) or you cannot measure your own performance.

---

## 2. The ML Lifecycle

```
   ┌──────────────────────────────────────────────────────────┐
   │                                                          ▼
BUSINESS → DATA → FEATURE → MODEL → EVALUATE → DEPLOY → MONITOR
 PROBLEM   COLLECT  ENG.     TRAIN                          │
   ▲                  ▲        ▲                            │
   │                  └────────┴──── retrain ───────────────┘
   └───────────────── redefine problem ────────────────────
```

### Maturity levels

| Level | Description |
|-------|-------------|
| **0 — Manual** | Notebooks, manual training, manual handoff. Fine for a first model, unsustainable after |
| **1 — ML pipeline automation** | Automated training pipeline, continuous training triggered by data or schedule |
| **2 — CI/CD automation** | Automated build, test, and deployment of the *pipeline itself*; automated rollback |

Be honest about where you are. Most teams should target level 1 and stop; level 2 is worth it
when you have many models or frequent retraining.

---

## 3. Data Engineering for ML

### Data versioning

Code versioning is solved; data versioning is not. Options: **DVC** (git-like for data),
**LakeFS**, **Delta Lake** / **Iceberg** (versioned table formats), or immutable
date-partitioned snapshots in object storage.

**Minimum bar:** every trained model records exactly which data snapshot produced it. Without
that, you cannot reproduce or debug anything.

### Data validation

Validate on every pipeline run — *before* training, and again at serving time.

| Check | Example |
|-------|---------|
| **Schema** | Expected columns, types, nullability |
| **Range** | `age` between 0 and 120 |
| **Distribution** | Mean/std within expected bounds vs a reference |
| **Volume** | Row count within ±20% of typical |
| **Freshness** | Latest timestamp within the expected lag |
| **Uniqueness** | No unexpected duplicate keys |
| **Referential** | Foreign keys resolve |

Tools: **Great Expectations**, **Pandera**, **Evidently**, **Deequ**, **dbt tests**.

### Feature stores

Solve two specific problems:

1. **Training/serving skew** — features computed one way in a training SQL query and another way
   in production Python. The feature store makes both read the *same* definition.
2. **Point-in-time correctness** — when building training data, each row must use only feature
   values as they were **at that row's timestamp**, not today's values. Getting this wrong is
   the most subtle and most common form of temporal leakage.

Architecture: an **offline store** (warehouse, for training) and an **online store**
(low-latency KV store like Redis/DynamoDB, for serving) fed from shared definitions.

Tools: Feast (open source), Tecton, Databricks/Vertex/SageMaker feature stores.

**Do you need one?** Not for a single batch model. Yes when multiple teams share features, or
you have real-time serving with non-trivial feature computation.

---

## 4. Experiment Tracking and Reproducibility

### What to log for every run

| Category | Items |
|----------|-------|
| **Code** | Git commit SHA, diff if dirty |
| **Data** | Dataset version/hash, row counts, split seeds |
| **Config** | Every hyperparameter |
| **Environment** | Python version, library versions, CUDA, hardware |
| **Metrics** | Train/val curves, final test metrics, per-slice metrics |
| **Artifacts** | Model weights, plots, confusion matrix, feature importances |
| **Metadata** | Run duration, cost, who ran it, why |

Tools: **MLflow** (open source, self-hostable), **Weights & Biases** (best UX), **Neptune**,
**Comet**, **DVC experiments**, or **TensorBoard** for the basics.

```python
import mlflow

with mlflow.start_run(run_name="lgbm-baseline"):
    mlflow.log_params({"lr": 0.05, "max_depth": 6, "n_estimators": 500})
    mlflow.log_metric("val_auc", 0.873)
    mlflow.log_artifact("plots/confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")
```

### Reproducibility checklist

- [ ] Seeds set for Python, NumPy, and the framework (`torch.manual_seed`, etc.)
- [ ] Data snapshot pinned and hashed
- [ ] Dependencies pinned (`requirements.txt` with `==`, or a lockfile)
- [ ] Environment containerized
- [ ] Preprocessing inside the pipeline artifact, not in a separate notebook
- [ ] Training script runnable from the CLI with no manual steps

> Exact GPU reproducibility is only approximate (nondeterministic kernels, atomic-add ordering).
> `torch.use_deterministic_algorithms(True)` gets you closer, at a performance cost. Aim for
> *statistical* reproducibility: the same setup gives you the same result within noise, and you
> know what that noise is.

---

## 5. Model Packaging and Serving

### Serialization formats

| Format | Pros | Cons |
|--------|------|------|
| **Pickle / joblib** | Trivial for sklearn | **Arbitrary code execution on load**; version-fragile |
| **ONNX** | Framework-agnostic, optimized runtimes | Not all ops supported |
| **TorchScript** | PyTorch, no Python at inference | PyTorch only |
| **SavedModel** | TF ecosystem standard | TF only |
| **Safetensors** | **Safe** (no code execution), fast, memory-mapped | Weights only, no graph |
| **GGUF** | Quantized LLMs for CPU/edge (llama.cpp) | LLM-specific |

> **Never unpickle a model file from an untrusted source.** `pickle.load` can execute arbitrary
> code. Use safetensors or ONNX where you can.

**Always package the full pipeline**, not just the estimator — the scaler, the encoder, and the
model must travel together, or you get training/serving skew.

```python
from sklearn.pipeline import Pipeline
import joblib

pipe = Pipeline([("scaler", scaler), ("model", model)])
joblib.dump(pipe, "model.joblib")   # ✅ one artifact, no skew
```

### Serving patterns

| Pattern | Latency | Use when |
|---------|---------|----------|
| **Batch (offline)** | Hours | Scoring all users nightly; recommendations precomputed |
| **Online (REST/gRPC)** | ms | Per-request predictions |
| **Streaming** | seconds | Kafka/Flink event processing |
| **Edge / on-device** | ms, offline capable | Mobile, IoT, privacy-sensitive |
| **Embedded in the app** | µs | Small models, no network hop |

**Batch is underrated.** If your use case tolerates staleness, precomputing predictions into a
key-value store is dramatically simpler, cheaper, and more reliable than online serving.

### A minimal serving API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, logging

app = FastAPI()
model = joblib.load("model.joblib")     # load once at startup, not per request

class Features(BaseModel):
    sqft: float = Field(gt=0, lt=100_000)
    bedrooms: int = Field(ge=0, le=20)

@app.post("/predict")
def predict(x: Features):
    try:
        pred = model.predict([[x.sqft, x.bedrooms]])[0]
    except Exception:
        logging.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference error")
    logging.info("prediction", extra={"features": x.dict(), "pred": float(pred)})
    return {"prediction": float(pred)}

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.2.0"}
```

**Non-negotiables for a production endpoint:** input validation (Pydantic), a health check,
structured logging of inputs *and* outputs (this is your monitoring data and your future
training data), a model version in the response, timeouts, and graceful degradation to a
fallback when the model fails.

### Optimizing inference

| Technique | Speedup |
|-----------|---------|
| Batching requests | Large — amortizes overhead |
| Quantization (INT8) | 2–4× |
| ONNX Runtime / TensorRT | 2–5× |
| Distillation into a smaller model | 5–50× |
| Caching frequent inputs | Unbounded |
| Removing Python from the hot path | 2–10× |

**Profile before optimizing.** Feature computation and network I/O frequently dominate model
inference time, and optimizing the model then buys you nothing.

---

## 6. Deployment Strategies

| Strategy | How | Risk | Use when |
|----------|-----|------|----------|
| **Shadow / dark launch** | New model runs alongside, predictions logged but not served | **None** | Always, as the first step |
| **Canary** | Route 1% → 5% → 25% → 100% | Low | Standard rollout |
| **A/B test** | Random split, measure business metrics | Low | You need statistical evidence of impact |
| **Blue-green** | Two full environments, switch traffic | Low | Instant rollback needed |
| **Multi-armed bandit** | Adaptively route more traffic to the winner | Low | Many variants, want to minimize regret |

**The correct sequence:** shadow first (validate that predictions are sane and latency is
acceptable with zero user risk), then canary, then A/B for the business metric, then full
rollout. Always keep the previous model deployable for instant rollback.

> **Offline metrics do not guarantee online improvement.** A model with better AUC can produce
> worse business outcomes — because of threshold effects, latency, feedback loops, or a
> mismatch between your loss and the actual decision. **A/B test the metric that matters.**

---

## 7. Monitoring

### The four layers

**1. Operational** — is it running?
Latency (p50/p95/p99), throughput, error rate, CPU/GPU/memory, uptime.

**2. Data quality** — is the input sane?
Missing-value rates, schema violations, out-of-range values, cardinality changes, volume.

**3. Model behaviour** — is it predicting sensibly?
Prediction distribution, confidence distribution, class balance of outputs, feature
distributions, rate of fallback/error paths.

**4. Business impact** — is it creating value?
Conversion, revenue, false-positive cost, user satisfaction, override rate by human reviewers.

> **Layers 1 and 4 are what people monitor; 2 and 3 are what catch model failure.** The gap
> between "the service is healthy" and "the model is right" is where silent failures live.

### The ground-truth delay problem

You usually can't compute accuracy in real time — labels arrive later (a loan defaults in 6
months; a fraud chargeback lands in 30 days) or never.

**Proxies for the interim:**
- Prediction distribution shift (see §8)
- Input feature drift
- Confidence/entropy distribution shifts
- Human override rates
- Downstream behavioural signals (click-through, complaint rate)
- Deliberate sampling: send a random slice for human labelling continuously

### Alerting

Alert on **user-visible or business-relevant** symptoms with clear ownership and a runbook.
Don't alert on every statistical wobble — a drift test on a large sample will flag
statistically-significant-but-meaningless changes daily, and alert fatigue is how real incidents
get missed. Use effect-size thresholds (PSI > 0.2), not just p-values.

---

## 8. Drift Detection

Recall the taxonomy from [Module 02 §9](02_ml_fundamentals.md#distribution-shift): covariate
shift, label shift, concept drift, domain shift.

### Statistical tests

| Test | Data type | Notes |
|------|-----------|-------|
| **Kolmogorov–Smirnov** | Continuous | Max distance between CDFs; over-sensitive at large $n$ |
| **Chi-squared** | Categorical | Compares frequency distributions |
| **Population Stability Index** | Both (binned) | **The industry standard** |
| **Wasserstein distance** | Continuous | Earth mover's distance; interpretable magnitude |
| **JS divergence** | Distributions | Symmetric, bounded |
| **Domain classifier** | Any | Train a model to distinguish old vs new data; AUC ≈ 0.5 means no drift |

**Population Stability Index:**

$$\text{PSI} = \sum_{i=1}^{B}(\text{actual}_i - \text{expected}_i)\ln\left(\frac{\text{actual}_i}{\text{expected}_i}\right)$$

| PSI | Interpretation |
|-----|---------------|
| < 0.1 | No significant shift |
| 0.1–0.2 | Moderate shift — investigate |
| > 0.2 | **Significant shift — act** |

### What to do about drift

Drift is a *signal*, not automatically a problem. Steps:

1. **Confirm it's real** — not a pipeline bug, an upstream schema change, or a seasonal pattern
   you already knew about. Most "drift alerts" are broken pipelines.
2. **Check whether performance actually degraded** — feature drift with stable performance may
   need no action.
3. **Identify the cause** — a new user segment, a competitor's launch, a product change, a data
   source change.
4. **Then respond** — retrain, add features, adjust thresholds, or (if the world changed
   fundamentally) redesign.

---

## 9. Retraining

### Triggers

| Trigger | Description | Best for |
|---------|-------------|----------|
| **Scheduled** | Daily/weekly/monthly | Predictable drift, simple to operate |
| **Performance-based** | Metric drops below a threshold | Requires timely labels |
| **Drift-based** | PSI exceeds a threshold | No labels needed |
| **Data volume** | Every N new labeled examples | Cold-start growth phase |
| **Manual** | On demand | Low-velocity domains |

**Start with scheduled retraining.** It's predictable, testable, and easy to reason about. Add
triggers once you understand your actual drift rate.

### Retraining strategy

| Approach | Notes |
|----------|-------|
| **Full retrain from scratch** | Simplest, most reliable — **the default** |
| **Warm start / incremental** | Faster; risks accumulating drift and hidden state |
| **Sliding window** | Train on the last N months — adapts fast, forgets rare patterns |
| **Weighted** | Recent data weighted higher; a good compromise |

**The retraining pipeline must be automatic and gated:**

```
new data → validate → train → evaluate against the CURRENT production model
        → if better on the primary metric AND no slice regressions → deploy to shadow
        → canary → promote
        → if worse → alert, keep the current model, don't silently degrade
```

**Automatic promotion without a gate is dangerous.** A poisoned or broken data batch will
happily train a worse model that sails into production. The comparison against the incumbent is
the safety mechanism.

---

## 10. Testing ML Systems

### The test pyramid for ML

| Level | Tests |
|-------|-------|
| **Unit** | Feature transforms, custom metrics, data utilities — deterministic, fast |
| **Data** | Schema, ranges, distributions, no-leakage assertions |
| **Model** | Shape correctness, overfit-a-tiny-batch, gradient checks, invariance tests |
| **Behavioural** | Task-level capabilities (see below) |
| **Integration** | The full pipeline end to end on a small fixture |
| **Performance** | Latency, memory, throughput under load |

### Behavioural testing (the CheckList framework)

Test **capabilities**, not just aggregate accuracy:

| Test type | Description | Example |
|-----------|-------------|---------|
| **Minimum functionality** | Simple cases that must work | "This is terrible" → negative sentiment |
| **Invariance** | Irrelevant changes shouldn't change the prediction | Changing a name shouldn't change a loan decision |
| **Directional** | Known changes should move the prediction predictably | Adding bedrooms shouldn't lower a house price |

These catch failures that aggregate metrics hide, and they double as **fairness tests** —
invariance under protected-attribute changes is exactly a fairness property.

### Useful assertions

```python
def test_overfits_single_batch():
    """If the model can't memorize 8 examples, there's a bug."""
    X, y = get_tiny_batch(n=8)
    model = build_model()
    for _ in range(200):
        train_step(model, X, y)
    assert loss(model, X, y) < 0.01

def test_no_target_leakage():
    """No feature should correlate near-perfectly with the target."""
    for col in X.columns:
        assert abs(np.corrcoef(X[col], y)[0, 1]) < 0.99, f"{col} looks like leakage"

def test_pipeline_is_serializable():
    joblib.dump(pipe, "/tmp/m.joblib")
    loaded = joblib.load("/tmp/m.joblib")
    np.testing.assert_allclose(pipe.predict(X_test), loaded.predict(X_test))

def test_serving_matches_training():
    """Training/serving skew guard."""
    np.testing.assert_allclose(
        training_features(raw_row), serving_features(raw_row), rtol=1e-6
    )
```

That last one is worth writing on day one. Training/serving skew is the most expensive bug class
in production ML, and it's trivially detectable with one test.

---

## 11. Cost, Scale, and Reliability

### Cost drivers

| Driver | Reduction strategy |
|--------|-------------------|
| **GPU inference** | Quantization, distillation, batching, autoscale to zero, right-size instances |
| **Training** | Spot/preemptible instances, checkpointing, mixed precision, early stopping |
| **Data storage** | Lifecycle policies, columnar formats (Parquet), compression |
| **Feature computation** | Cache, precompute, prune unused features |
| **LLM API calls** | Smaller models, shorter prompts, caching, routing ([Module 06 §11](06_nlp_and_llms.md#11-serving-and-inference-optimization)) |

**Measure cost per prediction and track it like a metric.** Teams routinely discover a model
costing more than the value it generates.

### Scaling

- **Horizontal** — more replicas behind a load balancer. Stateless inference makes this easy.
- **Vertical** — bigger machines. Simpler, has a ceiling.
- **Autoscaling** — on queue depth or request rate; **beware cold starts** with large models
  (loading a multi-GB model takes tens of seconds — keep a warm pool).
- **Async / queued** — for long-running inference, return a job ID and poll or webhook.

### Reliability patterns

| Pattern | Purpose |
|---------|---------|
| **Graceful degradation** | Fall back to a simpler model, a cached prediction, or a heuristic |
| **Circuit breaker** | Stop calling a failing dependency; fail fast |
| **Timeouts** | Never let a slow model hold a request forever |
| **Rate limiting** | Protect against traffic spikes |
| **Idempotency** | Safe retries |
| **Feature-flag the model** | Disable an ML path instantly without a deploy |

**Design the fallback before you deploy the model.** "What do we serve if the model is down or
returns garbage?" should have a concrete answer — usually the previous model, a cached value, or
the rule-based system the model replaced.

---

## 12. Responsible AI

### Fairness

| Definition | Criterion | Note |
|-----------|-----------|------|
| **Demographic parity** | $P(\hat y=1\mid A=a)$ equal across groups | Ignores real base-rate differences |
| **Equal opportunity** | Equal **TPR** across groups | Good when false negatives are the harm |
| **Equalized odds** | Equal TPR **and** FPR | Stricter |
| **Calibration within groups** | Predicted probabilities mean the same thing per group | Often what you want for risk scores |
| **Individual fairness** | Similar individuals get similar predictions | Hard to formalize |

> **Impossibility result:** when base rates genuinely differ across groups, calibration and
> equalized odds **cannot both hold** (Kleinberg et al., Chouldechova). You must choose which
> fairness criterion matters for your context — there is no universally fair model, and any
> vendor claiming otherwise is selling something.

**Practical process:** identify protected attributes and proxies for them (zip code proxies
race; first name proxies gender); measure metrics **per subgroup**, always; check worst-group
performance, not just the average; audit the training data for representation; document the
tradeoff you chose and why.

**Mitigations:** pre-processing (reweight/resample), in-processing (fairness constraints in the
objective), post-processing (group-specific thresholds — legally fraught in some jurisdictions,
check before using).

### Explainability

| Method | Scope | Notes |
|--------|-------|-------|
| **Coefficients** | Global | Linear models — read them directly |
| **Feature importance** | Global | Trees; biased toward high-cardinality features |
| **Permutation importance** | Global | Model-agnostic; unreliable with correlated features |
| **Partial dependence / ICE** | Global / local | Shows the marginal effect of a feature |
| **SHAP** | Both | Shapley values; theoretically grounded, additive, the current standard |
| **LIME** | Local | Local surrogate model; faster, less stable than SHAP |
| **Counterfactuals** | Local | "Increase income by $5k and you'd be approved" — the most *actionable* explanation |
| **Attention maps** | Local | Suggestive, but **attention is not explanation** — treat with care |

**SHAP** assigns each feature its Shapley value — the average marginal contribution over all
feature orderings — which uniquely satisfies efficiency, symmetry, dummy, and additivity:

$$\phi_i = \sum_{S\subseteq F\setminus\{i\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{i\}) - f(S)\big]$$

Exact computation is exponential; `TreeSHAP` computes it in polynomial time for tree ensembles,
which is why SHAP is practical for the models you'll actually deploy.

### Privacy and governance

**Privacy:** minimize data collection; anonymize and pseudonymize; be aware that **models
memorize** (LLMs can regurgitate training data, and membership-inference attacks are real);
consider differential privacy (DP-SGD) for sensitive data; federated learning where data can't
be centralized. Know your regulatory obligations (GDPR right to explanation and erasure, CCPA,
sector rules like HIPAA and the EU AI Act's risk tiers).

**Documentation:** **Model Cards** (intended use, out-of-scope uses, training data,
per-subgroup evaluation, limitations, ethical considerations) and **Datasheets for Datasets**
(collection process, composition, consent, maintenance). Write these — they force you to
articulate limitations you'd otherwise leave implicit, and they're increasingly a compliance
requirement.

**Human oversight:** for consequential decisions (credit, hiring, medical, criminal justice),
keep a human in the loop, provide explanations they can actually act on, monitor override rates
(a very low override rate may mean rubber-stamping, not accuracy), and offer an appeal path.

---

## Production readiness checklist

**Data**
- [ ] Versioned and reproducible
- [ ] Validation checks running on every pipeline execution
- [ ] Leakage audited
- [ ] Point-in-time correctness verified for time-based features

**Model**
- [ ] Beats a documented baseline
- [ ] Evaluated on a test set touched exactly once
- [ ] Per-subgroup metrics measured
- [ ] Calibrated if probabilities drive decisions
- [ ] Failure modes documented

**Engineering**
- [ ] Full pipeline packaged as one artifact
- [ ] Input validation on the endpoint
- [ ] Health check, structured logging, model version exposed
- [ ] Latency meets its SLO under realistic load
- [ ] Training/serving skew test passing
- [ ] Rollback tested, not just planned

**Operations**
- [ ] Monitoring across all four layers
- [ ] Drift detection configured with effect-size thresholds
- [ ] Alerts route to a named owner with a runbook
- [ ] Retraining pipeline automated and gated on a comparison to production
- [ ] Fallback behaviour defined and tested
- [ ] Cost per prediction tracked

**Governance**
- [ ] Model card written
- [ ] Fairness assessed and the chosen tradeoff documented
- [ ] Privacy review complete
- [ ] Human oversight in place for consequential decisions

---

## Self-check

1. Why do ML systems fail silently, and what does that imply for monitoring?
2. What two problems does a feature store solve?
3. Explain point-in-time correctness and how violating it leaks the future.
4. Why should the retraining pipeline compare against the current production model?
5. Compute and interpret a PSI of 0.35.
6. Why can't a model be both calibrated and satisfy equalized odds when base rates differ?
7. Your model has better offline AUC but worse online conversion. Give three explanations.
8. Write the test that catches training/serving skew.

---

**Next:** [10 — Glossary →](10_glossary.md)
