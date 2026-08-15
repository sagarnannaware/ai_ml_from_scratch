# 06 — NLP and Large Language Models

> From tokenization to production LLM applications: how these models are built, adapted,
> aligned, retrieved-augmented, and served.

📄 Companion code: `python/08_nltk_spacy_nlp.py`, `python/09_huggingface_transformers_llm.py`

---

## Table of contents

1. [Classical NLP foundations](#1-classical-nlp-foundations)
2. [Tokenization](#2-tokenization)
3. [Word embeddings](#3-word-embeddings)
4. [Language modelling](#4-language-modelling)
5. [Pretraining](#5-pretraining)
6. [Adapting a pretrained model](#6-adapting-a-pretrained-model)
7. [Alignment](#7-alignment)
8. [Retrieval-Augmented Generation (RAG)](#8-retrieval-augmented-generation-rag)
9. [Tool use and agents](#9-tool-use-and-agents)
10. [Evaluating LLM systems](#10-evaluating-llm-systems)
11. [Serving and inference optimization](#11-serving-and-inference-optimization)
12. [Limitations and risks](#12-limitations-and-risks)

---

## 1. Classical NLP Foundations

Still worth knowing — these are strong baselines, and they're often the right answer when
latency or cost matters.

### The preprocessing pipeline

| Step | What it does | Notes |
|------|-------------|-------|
| **Normalization** | Lowercase, strip accents, unicode NFKC | May destroy signal (ALL CAPS in spam) |
| **Tokenization** | Split into units | See §2 |
| **Stopword removal** | Drop "the", "is", "at" | Helps bag-of-words; **harmful** for transformers |
| **Stemming** | Chop to a crude root: "running" → "run" | Porter stemmer; fast, produces non-words |
| **Lemmatization** | Map to a dictionary form: "better" → "good" | Slower, linguistically correct |
| **POS tagging** | Label each token's part of speech | Noun, verb, adjective… |
| **NER** | Extract named entities | Person, organization, location, date |
| **Dependency parsing** | Grammatical structure between words | Who did what to whom |

### Text representations

**Bag of Words (BoW):** a vector of word counts. Loses all order.

**TF-IDF:** weights terms by how distinctive they are to a document:

$$\text{TF-IDF}(t,d) = \text{tf}(t,d)\times\log\frac{N}{1+\text{df}(t)}$$

**n-grams:** contiguous sequences of $n$ tokens. Bigrams recover a little word order ("not good"
vs "good") at the cost of a much larger vocabulary.

**BM25** — the ranking function behind most keyword search engines (Elasticsearch, Lucene). An
improvement over TF-IDF with term-frequency saturation and document-length normalization:

$$\text{BM25}(q,d) = \sum_{t\in q}\text{IDF}(t)\cdot\frac{\text{tf}(t,d)\cdot(k_1+1)}{\text{tf}(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

with $k_1\approx1.5$, $b\approx0.75$. **Still essential** — hybrid BM25 + vector search
consistently beats vector search alone in RAG.

---

## 2. Tokenization

Neural models operate on integer IDs, so text must be split into units and mapped to a
vocabulary.

| Granularity | Vocabulary | Sequence length | Problem |
|-------------|-----------|-----------------|---------|
| **Character** | ~100 | Very long | No semantic units; slow |
| **Word** | 100k–1M+ | Short | Out-of-vocabulary words; huge embedding table |
| **Subword** | 30k–200k | Medium | **The right answer** |

### Byte-Pair Encoding (BPE)

The dominant algorithm (GPT, Llama, most modern LLMs).

**Training:**
```
1. Start with a vocabulary of individual characters (or bytes)
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol; record the merge rule
4. Repeat until the vocabulary reaches the target size
```

**Encoding:** apply the learned merge rules in order to new text.

Result: frequent words become single tokens; rare words decompose into meaningful pieces
("tokenization" → "token" + "ization"). Nothing is ever out-of-vocabulary.

**Byte-level BPE** (GPT-2 onward) operates on raw UTF-8 bytes, so **any** string is
representable — no `<UNK>` token can ever be needed.

### Other algorithms

| Algorithm | Used by | Idea |
|-----------|---------|------|
| **WordPiece** | BERT | Like BPE, but merges the pair that most increases corpus likelihood, not raw frequency |
| **Unigram LM** | T5, ALBERT | Start with a large vocabulary, iteratively *remove* tokens that cost the least likelihood |
| **SentencePiece** | Llama, T5 | Language-agnostic wrapper; treats the input as a raw stream (no pre-tokenization on spaces), so it works for Chinese/Japanese |

### Practical consequences

- **1 token ≈ 4 characters ≈ 0.75 words** in English. Non-English text and code use far more
  tokens per character — an under-appreciated cost and equity issue.
- **Costs and context limits are in tokens**, so tokenization directly affects your bill.
- **Numbers tokenize badly** — "1234" may become "12"+"34", which is part of why LLMs struggle
  with arithmetic.
- **The famous "how many r's in strawberry" failure** is a tokenization artifact: the model
  never sees individual letters, only subword chunks.
- Special tokens: `[CLS]`, `[SEP]`, `[MASK]`, `<|endoftext|>`, `<pad>`.

---

## 3. Word Embeddings

Dense vector representations where geometric proximity encodes semantic similarity.

### Word2Vec (Mikolov et al., 2013)

Two architectures:
- **CBOW:** predict a word from its surrounding context.
- **Skip-gram:** predict the context from the word. Better for rare words.

Skip-gram objective:

$$\max\;\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\le j\le c,\,j\ne0}\log p(w_{t+j}\mid w_t)$$

with the softmax

$$p(w_O\mid w_I) = \frac{\exp(\mathbf{v}'^\top_{w_O}\mathbf{v}_{w_I})}{\sum_{w=1}^{V}\exp(\mathbf{v}'^\top_w\mathbf{v}_{w_I})}$$

That denominator sums over the entire vocabulary — far too expensive. **Negative sampling**
replaces it with a binary classification against $k$ random "negative" words:

$$\log\sigma(\mathbf{v}'^\top_{w_O}\mathbf{v}_{w_I}) + \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}\left[\log\sigma(-\mathbf{v}'^\top_{w_i}\mathbf{v}_{w_I})\right]$$

**The famous property:** $\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen})$.
Linear structure in the embedding space captures analogies.

### GloVe
Factorizes the global word co-occurrence matrix rather than using local windows. Combines
count-based and predictive approaches.

### Contextual embeddings — the key advance

Word2Vec/GloVe give each word **one fixed vector**, so "bank" (river) and "bank" (money) share
a representation. **ELMo, BERT, and every transformer since** produce embeddings that depend on
the surrounding sentence — the same word gets different vectors in different contexts. This was
the breakthrough that made modern NLP work.

### Sentence embeddings

For retrieval and similarity you need one vector per document, not per token.

| Approach | Notes |
|----------|-------|
| Mean-pool token embeddings | Cheap baseline; surprisingly weak with raw BERT |
| **Sentence-BERT (SBERT)** | Siamese network fine-tuned with a contrastive objective — **do this, not raw BERT** |
| Commercial embedding APIs | Strong, easy, per-token cost |
| Open models (E5, BGE, GTE, Nomic) | Competitive; self-hostable |

**Similarity:** cosine similarity (§1.3 of [Module 01](01_math_foundations.md#13-norms-measuring-size)).
Normalize the vectors and cosine similarity becomes a plain dot product — which is what vector
databases index.

---

## 4. Language Modelling

### The objective

A language model assigns probabilities to sequences. By the chain rule of probability:

$$P(w_1,\dots,w_T) = \prod_{t=1}^{T}P(w_t\mid w_1,\dots,w_{t-1})$$

Training minimizes the negative log-likelihood over a corpus:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(w_t\mid w_{<t};\theta)$$

**That's it.** Every autoregressive LLM — GPT, Llama, Claude — is trained on this one objective.
It is cross-entropy loss on next-token prediction, which is self-supervised: the labels come
free from the text itself.

**Perplexity:** $\text{PPL} = \exp(\mathcal{L})$. Lower is better.

### Decoding strategies

Given the model's next-token distribution, how do you pick a token?

| Strategy | How | Character |
|----------|-----|-----------|
| **Greedy** | $\arg\max_w P(w)$ | Deterministic; repetitive and dull |
| **Beam search** | Keep the $k$ best partial sequences | Good for translation; bland for open-ended text |
| **Pure sampling** | Sample from the full distribution | Diverse; often incoherent |
| **Temperature** | Sample from $\text{softmax}(z/T)$ | The main creativity dial |
| **Top-k** | Sample from the $k$ most likely tokens | Fixed cutoff regardless of confidence |
| **Top-p (nucleus)** | Sample from the smallest set whose cumulative probability $\ge p$ | **Adapts to confidence — the standard** |
| **Min-p** | Threshold relative to the top token's probability | Newer, robust at high temperature |

**Temperature:**

$$P_T(w_i) = \frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$$

- $T\to0$: approaches greedy (deterministic, focused).
- $T=1$: the model's raw distribution.
- $T>1$: flatter distribution, more random.

**Practical defaults:** factual/extraction tasks → $T=0$; general chat → $T\approx0.7$,
top-p $=0.9$; creative writing → $T\approx1.0$.

**Repetition penalties:** divide the logits of already-generated tokens by a factor (~1.1), or
apply a frequency/presence penalty. Necessary because greedy-ish decoding loves loops.

---

## 5. Pretraining

### Objectives

| Objective | Model family | Description |
|-----------|-------------|-------------|
| **Causal LM (CLM)** | GPT, Llama | Predict the next token given all previous ones |
| **Masked LM (MLM)** | BERT | Mask ~15% of tokens; predict them from bidirectional context |
| **Span corruption** | T5 | Mask contiguous spans; generate them |
| **Next Sentence Prediction** | BERT (original) | Do these two sentences follow? *Later shown unhelpful and dropped* |
| **Replaced token detection** | ELECTRA | Discriminate real vs generated tokens — much more sample-efficient |
| **Contrastive** | CLIP, SimCSE | Pull matched pairs together, push mismatched apart |

### Scaling laws

Model loss follows a power law in compute, data, and parameters (Kaplan et al. 2020; Hoffmann
et al. 2022):

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

where $N$ = parameters, $D$ = training tokens, and $L_\infty$ is the irreducible entropy of
natural language.

**Chinchilla-optimal scaling:** for a fixed compute budget, $N$ and $D$ should scale
*equally* — roughly **20 tokens per parameter**. The earlier generation of models (GPT-3 and
peers) was badly under-trained on data relative to size. In practice, models today are often
trained well past Chinchilla-optimal because **inference cost** dominates over training cost
for a deployed model: a smaller model trained longer is cheaper to serve forever.

**Compute estimate:** training FLOPs $\approx 6ND$ (forward ≈ $2ND$, backward ≈ $4ND$).

### Emergent abilities
Some capabilities (multi-step arithmetic, chain-of-thought reasoning, instruction following)
appear abruptly beyond a scale threshold rather than improving smoothly. Whether these are
genuinely discontinuous or an artifact of discontinuous metrics is actively debated — with good
evidence that smoother metrics show smoother curves.

### The data pipeline
Deduplication (near-duplicates hurt a lot), quality filtering (classifier-based), toxicity and
PII filtering, benchmark decontamination, and domain mixing (web, code, books, academic
papers). **Data quality determines model quality more than architecture does.**

---

## 6. Adapting a Pretrained Model

Ordered from cheapest to most expensive:

| Method | Trains weights? | Data needed | Cost | Use when |
|--------|----------------|-------------|------|----------|
| **Prompt engineering** | No | 0 | ~0 | **Always try first** |
| **Few-shot / in-context learning** | No | 2–50 examples | Token cost per call | Format/style steering |
| **RAG** | No | A document corpus | Retrieval infra | The model needs *knowledge* it lacks |
| **LoRA / PEFT** | Adapters only | 100s–10k examples | 1 GPU, hours | Consistent behaviour/format/domain |
| **Full fine-tuning** | All | 10k+ | Many GPUs | Deep domain adaptation |
| **Continued pretraining** | All | Billions of tokens | Very large | A new language or specialized domain |

> **The decision rule that saves the most money:** if the model *doesn't know something*, use
> RAG. If the model *doesn't behave how you want*, fine-tune. Fine-tuning is a poor way to
> inject facts (it's lossy, and you retrain on every update); retrieval is a poor way to change
> style. Most teams reach for fine-tuning when they needed retrieval.

### Prompt engineering

| Technique | Description |
|-----------|-------------|
| **Zero-shot** | Just ask |
| **Few-shot** | Include input/output examples in the prompt |
| **Chain-of-thought (CoT)** | "Let's think step by step" — dramatically improves multi-step reasoning |
| **Self-consistency** | Sample several CoT paths, take the majority answer |
| **ReAct** | Interleave reasoning traces with tool actions |
| **Structured output** | Request JSON / a schema; use constrained decoding to guarantee validity |
| **Role and context setting** | System prompt defines the persona, constraints, and format |
| **Decomposition** | Break a hard task into a chain of simpler calls |

**Chain-of-thought works** because it gives the model more forward passes — more serial
computation — to reach an answer, effectively letting it "use scratch space." A transformer has
fixed depth per token, so writing intermediate steps is genuinely extra computation.

### Supervised Fine-Tuning (SFT)

Train on (instruction, response) pairs with standard cross-entropy loss, usually masking the
loss on the prompt tokens so the model is only scored on generating the response.

Data quality dominates quantity — **1,000 excellent examples beat 100,000 mediocre ones**
(the LIMA result). Curate carefully.

### LoRA (Low-Rank Adaptation)

**The key insight:** the weight *update* during fine-tuning has low intrinsic rank. So don't
learn a full $\Delta W$; learn its factorization.

Freeze $W_0\in\mathbb{R}^{d\times k}$ and learn:

$$W = W_0 + \Delta W = W_0 + BA, \qquad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k},\; r \ll \min(d,k)$$

The forward pass becomes:

$$\mathbf{h} = W_0\mathbf{x} + \frac{\alpha}{r}BA\mathbf{x}$$

**Parameter count:** $r(d+k)$ instead of $dk$. For $d=k=4096$ and $r=8$: 65,536 trainable
parameters instead of 16.8M — **a 256× reduction.**

**Initialization:** $A\sim\mathcal{N}(0,\sigma^2)$, $B=0$. Starting with $B=0$ means
$\Delta W = 0$ initially, so training begins exactly at the pretrained model — no disruption.

| Hyperparameter | Typical | Notes |
|----------------|---------|-------|
| `r` (rank) | 8–64 | Higher = more capacity; 16 is a good default |
| `alpha` | $2r$ | Scaling factor; often kept at $2\times r$ |
| `target_modules` | `q_proj`, `v_proj` (minimum); all linear layers (better) | More targets = better quality |
| `dropout` | 0.05–0.1 | |

**Why it's practical:** memory drops enormously (no optimizer state for frozen weights),
adapters are tiny (a few MB) so you can host many task-specific adapters over one base model,
and at inference you can **merge** $BA$ into $W_0$ for zero added latency.

**QLoRA:** quantize the frozen base model to 4-bit (NF4) and train LoRA adapters on top in
bf16. This is what allows fine-tuning a 65B model on a single 48 GB GPU. Adds 4-bit
double-quantization and paged optimizers to manage memory spikes.

**Other PEFT methods:** prefix tuning, prompt tuning (learn soft prompt embeddings), adapters
(small bottleneck layers between blocks), IA³ (learned rescaling vectors), DoRA (decomposes
magnitude and direction).

### Catastrophic forgetting
Fine-tuning on a narrow task degrades general capability. Mitigate with lower learning rates,
fewer epochs (1–3 is usually right), LoRA (which limits how far weights can move), and mixing
in general-purpose data.

---

## 7. Alignment

Pretrained models predict likely text; they don't inherently follow instructions or refuse
harmful requests. Alignment closes that gap.

### The three-stage pipeline

```
1. PRETRAIN  → a base model that predicts text
2. SFT       → an instruction-following model
3. RLHF/DPO  → a model aligned with human preferences
```

### RLHF (Reinforcement Learning from Human Feedback)

**Step 1 — collect preferences.** Humans rank model outputs: given a prompt $x$, they mark
response $y_w$ (winner) as better than $y_l$ (loser).

**Step 2 — train a reward model** using the Bradley–Terry preference model:

$$P(y_w \succ y_l\mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\right]$$

**Step 3 — optimize the policy with PPO**, keeping it close to the SFT reference so it doesn't
drift into reward-hacking gibberish:

$$\max_\theta\;\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta}\big[r_\phi(x,y)\big] - \beta\,D_{\text{KL}}\big(\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big)$$

The KL penalty is doing essential work: without it, the policy finds adversarial inputs that
maximize the imperfect reward model while producing terrible text (**reward hacking**).

See [Module 08 §9](08_reinforcement_learning.md#9-ppo-proximal-policy-optimization) for PPO itself.

### DPO (Direct Preference Optimization)

**The insight:** you can solve for the optimal policy of the KL-constrained RLHF objective in
closed form, then substitute it back — eliminating the reward model and the RL loop entirely.
The reward is implicitly represented by the policy itself:

$$r(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$$

Substituting into the Bradley–Terry loss makes the intractable partition function $Z(x)$
cancel, leaving a simple classification loss:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

**Just supervised learning on preference pairs** — no reward model, no sampling, no RL
instability. This is why DPO has largely displaced PPO for open-source alignment.

**Variants:** IPO (fixes DPO's overfitting to deterministic preferences), KTO (works with
binary good/bad labels instead of pairs), ORPO (merges SFT and alignment in one stage),
GRPO (group-relative, used for reasoning models).

**RLAIF / Constitutional AI:** replace human preference labels with AI-generated ones guided by
an explicit set of written principles. Scales far better than human labelling.

---

## 8. Retrieval-Augmented Generation (RAG)

Give the model relevant documents at inference time instead of hoping it memorized them.

### Why RAG

| Problem | How RAG helps |
|---------|--------------|
| Knowledge cutoff | Retrieve current documents |
| Hallucination | Ground answers in real sources |
| Private/proprietary data | Never in the training set |
| Attribution | Cite the retrieved source |
| Cost of updating | Update the index, not the model |

### The pipeline

```
INDEXING (offline):
  documents → chunk → embed → store in a vector DB (+ keyword index)

QUERY (online):
  question → embed → retrieve top-k → rerank → build prompt → generate → cite
```

### Chunking — where most RAG systems are won or lost

| Strategy | Description |
|----------|-------------|
| **Fixed size** | $N$ tokens with overlap (e.g. 512 with 50) — simple baseline |
| **Recursive character** | Split on paragraph → sentence → word boundaries in order |
| **Semantic** | Split where consecutive-sentence embedding similarity drops |
| **Document-structure** | Split on markdown headers, sections, code functions |
| **Small-to-big** | Embed small chunks for retrieval precision, return the larger parent for context |
| **Contextual retrieval** | Prepend an LLM-generated summary of the document to each chunk before embedding — substantially reduces retrieval failures |

**Rules of thumb:** 200–800 tokens per chunk, 10–20% overlap, and **never split a table, code
block, or list mid-structure.** Attach metadata (source, title, date, section) to every chunk —
you need it for filtering and citation.

### Retrieval

| Method | Strength | Weakness |
|--------|----------|----------|
| **Dense (embeddings)** | Semantic similarity; handles paraphrase | Misses exact terms, IDs, rare names |
| **Sparse (BM25)** | Exact keyword matching; no training | No semantic understanding |
| **Hybrid** | **Both — use this** | Requires fusing two score scales |

**Reciprocal Rank Fusion (RRF)** — the standard way to merge two ranked lists without
normalizing incompatible scores:

$$\text{RRF}(d) = \sum_{r\in\text{retrievers}}\frac{1}{k + \text{rank}_r(d)}, \qquad k\approx60$$

**Vector indexes:** exact search is $O(nd)$; approximate nearest neighbour (ANN) is what
production uses.

| Index | Idea | Tradeoff |
|-------|------|----------|
| **HNSW** | Navigable small-world graph | Fast, high recall, memory-hungry — **the default** |
| **IVF** | Cluster into cells, search a few | Memory-efficient; tune `nprobe` |
| **PQ** | Product quantization compresses vectors | Big memory savings, some accuracy loss |
| **ScaNN / DiskANN** | Anisotropic quantization / on-disk | Scale |

**Reranking:** retrieve top-50 with a fast bi-encoder, then rescore with a **cross-encoder**
(which reads query and document *together* and is far more accurate but too slow to run over
the whole corpus) and keep the top-5. **This is the single highest-ROI improvement to a
mediocre RAG system.**

### Advanced patterns

| Pattern | What it does |
|---------|-------------|
| **Query rewriting** | Turn conversational input into a standalone search query |
| **Multi-query / RAG-fusion** | Generate several query variants, retrieve for each, fuse |
| **HyDE** | Generate a *hypothetical answer*, embed that, and search with it |
| **Self-RAG / CRAG** | The model decides whether to retrieve, and grades retrieved chunks |
| **GraphRAG** | Build a knowledge graph over entities for multi-hop questions |
| **Agentic RAG** | An agent iteratively searches, reads, and refines |

### Evaluating RAG — measure the stages separately

| Stage | Metrics |
|-------|---------|
| **Retrieval** | Recall@k, Precision@k, MRR, NDCG, hit rate |
| **Generation** | Faithfulness (is every claim supported by the context?), answer relevance |
| **End-to-end** | Correctness vs a gold answer, citation accuracy, refusal rate |

**Diagnose in order:** if the right chunk isn't in the retrieved set, no prompt engineering will
save you. Measure retrieval recall *first*. Most "the LLM hallucinated" reports are actually
retrieval failures.

---

## 9. Tool Use and Agents

### Tool / function calling

The model outputs a structured call, your code executes it, and the result is fed back:

```
user: "What's the weather in Tokyo?"
model → {"tool": "get_weather", "args": {"city": "Tokyo"}}
system executes → {"temp": 22, "condition": "cloudy"}
model → "It's 22°C and cloudy in Tokyo."
```

**Design rules for tools:** few, well-named, with precise descriptions and typed schemas;
return structured, compact results; return errors as informative text the model can recover
from; and make them idempotent where possible.

### The agent loop

```
while not done:
    thought = model.think(state)
    action  = model.choose_tool(thought)
    result  = execute(action)
    state   = state + [thought, action, result]
```

**Patterns:** ReAct (reason + act interleaved), Plan-and-Execute, Reflexion (self-critique and
retry), multi-agent (specialized roles), tree search over action sequences.

### Where agents actually break

| Failure | Mitigation |
|---------|-----------|
| Compounding errors over long horizons | Fewer steps; checkpoints; verification between steps |
| Infinite loops | Step limits, loop detection, timeouts |
| Cost explosion | Token budgets, cheaper models for routine steps |
| Getting stuck | Explicit failure paths; escalate to a human |
| **Prompt injection via tool results** | Treat all retrieved/tool content as untrusted data, never as instructions; sandbox execution; require confirmation for consequential actions |

> **Prompt injection is the defining security problem of agentic systems.** Any content the
> model reads — a web page, a document, an API response, an email — can contain instructions.
> There is no complete fix today. Architect so that a successful injection can't do irreversible
> damage: least privilege on tools, human confirmation for writes/sends/payments, and strict
> separation between the trusted instruction channel and untrusted data.

---

## 10. Evaluating LLM Systems

The hardest part of shipping LLM applications, and the part most teams skip.

### Automatic metrics (limited but cheap)

| Metric | Measures | Caveat |
|--------|----------|--------|
| **Perplexity** | Language modelling quality | Says nothing about task performance |
| **BLEU** | n-gram precision vs reference | Translation; poor for open-ended text |
| **ROUGE** | n-gram recall vs reference | Summarization; rewards copying |
| **METEOR** | Includes synonyms/stems | Better than BLEU, still shallow |
| **BERTScore** | Embedding similarity to a reference | Semantic, but needs a reference |
| **Exact match / F1** | Extractive QA | Only for constrained outputs |
| **pass@k** | Code that passes tests | **Excellent** — objective and meaningful |

### LLM-as-judge

Use a strong model to grade outputs against a rubric. Practical and widely used, with real
caveats: position bias (favours the first option — always randomize order), verbosity bias
(favours longer answers), self-preference bias, and poor calibration on subtle correctness.
**Validate the judge against human labels** on a sample before trusting it.

### Building an eval suite (do this before optimizing anything)

1. Collect 50–200 real, representative inputs — including the hard and adversarial ones.
2. Define what "correct" means for each, concretely.
3. Mix graders: exact/programmatic where possible, LLM-judge where not, humans for a sample.
4. Version it, and **run it in CI on every prompt or model change.**
5. Track regressions per category, not just an aggregate score.

> **Without an eval suite you are not engineering, you are vibing.** Prompt changes that
> obviously improve one case routinely break three others. This is the single biggest
> difference between demo-quality and production-quality LLM work.

### Common benchmarks
MMLU (knowledge), GSM8K/MATH (math), HumanEval/MBPP (code), HellaSwag (commonsense),
TruthfulQA (truthfulness), MT-Bench/Arena (chat quality), GPQA (graduate-level reasoning),
SWE-bench (real software engineering). Beware **contamination** — public benchmarks leak into
training data, so a high score may reflect memorization.

---

## 11. Serving and Inference Optimization

### The two phases

| Phase | What happens | Bound by |
|-------|-------------|----------|
| **Prefill** | Process the whole prompt in parallel | **Compute** |
| **Decode** | Generate one token at a time | **Memory bandwidth** |

Decoding is memory-bound because generating each token requires reading all model weights from
HBM while doing very little arithmetic. This shapes every optimization below.

### KV cache

During autoregressive generation, the keys and values of previous tokens don't change — so
cache them instead of recomputing. This turns generation from $O(L^2)$ to $O(L)$ per token.

**Cache size:**

$$\text{bytes} = 2 \times L \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{precision} \times \text{batch}$$

(The 2 is for K and V.) This grows linearly with context length and batch size, and quickly
becomes the **dominant memory consumer** — which is why **Grouped-Query Attention** (heads share
K/V) and **PagedAttention** (vLLM's virtual-memory-style paging that eliminates fragmentation)
matter so much.

### Quantization

Store weights in fewer bits.

| Format | Bits | Quality loss | Notes |
|--------|------|-------------|-------|
| FP16 / BF16 | 16 | None (baseline) | Standard |
| INT8 | 8 | Minimal | ~2× memory saving |
| **INT4 / NF4** | 4 | Small but real | ~4× saving; GPTQ, AWQ, bitsandbytes |
| INT2 | 2 | Severe | Research |

**Post-training quantization (PTQ)** is applied after training (GPTQ, AWQ — fast, no retraining).
**Quantization-aware training (QAT)** simulates quantization during training for better quality
at low bit widths.

**Rule of thumb: a 4-bit larger model usually beats a 16-bit smaller one** at the same memory
budget.

### Throughput techniques

| Technique | Benefit |
|-----------|---------|
| **Continuous batching** | Insert new requests as others finish instead of waiting for the whole batch — 2–10× throughput. The core of vLLM/TGI |
| **PagedAttention** | Near-zero KV cache waste |
| **FlashAttention** | Faster, memory-efficient exact attention |
| **Speculative decoding** | A small draft model proposes $k$ tokens; the big model verifies them in one pass — 2–3× faster with *identical* output distribution |
| **Prompt/prefix caching** | Reuse the KV cache for a shared system prompt across requests |
| **Tensor parallelism** | Split the model across GPUs for lower latency |

### Metrics that matter

| Metric | Meaning |
|--------|---------|
| **TTFT** (time to first token) | Prefill latency — dominates *perceived* responsiveness |
| **TPOT / ITL** | Time per output token — determines streaming smoothness |
| **Throughput** | Tokens/sec across all concurrent requests |
| **Cost per 1M tokens** | The number your finance team cares about |

**Latency ≠ throughput.** Larger batches raise throughput and *hurt* per-request latency. Decide
which you're optimizing before tuning anything.

### Cost reduction, in order of impact
1. **Use a smaller model** — try the smallest that passes your evals, then move up only if needed.
2. **Shorten prompts** — you pay per token on every single call.
3. **Cache** — exact-match caching plus prompt-prefix caching.
4. **Route by difficulty** — cheap model first, escalate only on low confidence.
5. **Batch** offline workloads.
6. **Self-host** only when volume justifies the engineering (the crossover is higher than people expect).

---

## 12. Limitations and Risks

| Limitation | Why it happens | Mitigation |
|------------|---------------|------------|
| **Hallucination** | The model optimizes fluency/likelihood, not truth; it has no ground-truth signal at inference | RAG with citations, constrained outputs, verification passes, explicit "I don't know" training, human review for high-stakes uses |
| **Knowledge cutoff** | Training ends at a date | RAG, tools, web search |
| **Arithmetic and counting** | Tokenization plus fixed compute per token | Give it a calculator/code tool |
| **Context limits & "lost in the middle"** | Attention degrades over very long inputs; information mid-context is used less reliably | Put critical content at the start or end; retrieve less, better |
| **Prompt injection** | No robust separation between instructions and data | Least privilege, sandboxing, human confirmation, treat all external content as untrusted |
| **Bias** | Training data reflects societal bias | Evaluate across subgroups, curate data, red-team |
| **Non-determinism** | Sampling; also floating-point nondeterminism even at $T=0$ | $T=0$ + fixed seed gets you close, not to a guarantee |
| **Sycophancy** | Preference training rewards agreement | Ask for critique explicitly; avoid leading questions |
| **Cost/latency** | Large models are expensive | See §11 |

**Calibrate your expectations:** LLMs are extraordinary at transformation, extraction,
summarization, drafting, classification, and code generation. They are unreliable as
*databases of fact* and as *arithmetic engines*. Design systems that use their strengths and
route around their weaknesses.

---

## Self-check

1. Why does BPE eliminate out-of-vocabulary tokens entirely?
2. Write the LM objective and explain why it's self-supervised.
3. Explain top-p sampling and why it beats top-k.
4. What does Chinchilla scaling say, and why do labs deliberately over-train past it?
5. Compute LoRA's trainable parameter count for $d=k=4096$, $r=16$.
6. Why is $B$ initialized to zero in LoRA?
7. Derive intuitively why DPO removes the need for a reward model.
8. Your RAG system gives wrong answers. What do you measure first, and why?
9. Why is decoding memory-bandwidth-bound while prefill is compute-bound?
10. Why does speculative decoding not change the output distribution?

---

**Next:** [07 — Computer Vision →](07_computer_vision.md)
