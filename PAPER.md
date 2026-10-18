# Tether: A Reproducible Benchmark for the Cost/Accuracy Frontier of RAG Faithfulness Detection

**Nishit Patel**
Arizona State University

> Working draft. All numbers in this paper are produced by
> `python scripts/generate_report.py` from the bundled dataset, so every figure
> is reproducible from a single command.

## Abstract

Retrieval-augmented generation (RAG) systems can produce answers that are not
supported by their retrieved context. Detecting these hallucinations is
important in production, and a strong large language model (LLM) judge can do it
well, but running such a judge on every answer is slow and expensive. In
practice teams want cheaper detectors. We present Tether, a small, fully
reproducible benchmark that measures how well low-cost faithfulness detectors
separate grounded answers from hallucinations, and how closely they approximate
an LLM judge. Tether contributes an error-type stratified dataset, a set of
baselines spanning three cost tiers (free lexical heuristics, a local NLI model,
and a paid LLM judge), agreement metrics against the judge, and a cost/accuracy
frontier that makes the trade-off explicit. On the bundled sample, free
detectors already clear the trivial floor by a wide margin, but they lose recall
specifically on hallucinations that reuse the context vocabulary, which is where
a semantic model or an LLM judge earns its cost.

## 1. Introduction

A RAG system retrieves context and asks a model to answer from it. When the
answer asserts something the context does not support, the system has
hallucinated. Catching that automatically is valuable, and the most reliable
automatic approach is to ask a strong LLM to judge faithfulness. That judge is
accurate but costly to run at scale, so teams reach for cheaper proxies such as
lexical overlap or embedding similarity without a clear picture of what accuracy
they give up.

Tether makes that picture concrete. It asks one question: how closely does a
cheap faithfulness detector track an LLM judge, and what does each step up the
cost ladder buy? The benchmark is deliberately small and dependency-light so
that its core baselines and metrics run offline and deterministically, which
makes the results easy to reproduce and extend.

## 2. Task

Each example is a triple of (question, context, answer) with a gold label:
`faithful` if every claim in the answer is supported by the context, and
`hallucinated` otherwise. A detector maps a triple to a faithfulness score in
[0, 1], where higher means more supported. The harness converts scores to
hallucination predictions with a threshold and reports precision, recall, F1,
accuracy, and AUROC. The positive class is the hallucinated answer, since that
is what we want to catch.

## 3. Dataset

The bundled dataset is 36 hand-written examples, balanced 18 faithful and 18
hallucinated, across general-knowledge domains. Every hallucinated example is
annotated with an error type, and every faithful example with a support type:

- **numeric**: a wrong number, date, or count.
- **entity_swap**: a wrong entity that reuses the context vocabulary.
- **unsupported_addition**: an extra claim the context does not make.
- **contradiction**: a statement that directly contradicts the context.
- **direct** / **paraphrase**: faithful answers that copy or restate the context.

The dataset is intentionally small and curated so it ships in the repository and
the benchmark runs with no download. The error-type annotations are what let the
benchmark report where detectors fail rather than only how often.

## 4. Detectors

Tether groups detectors into three cost tiers:

- **free**: `majority_constant` (a constant-score floor), `lexical_overlap`
  (fraction of answer content tokens present in the context), `novel_token`
  (penalizes answer tokens absent from both context and question), and
  `tfidf_cosine` (TF-IDF cosine between answer and context).
- **model**: `nli_entailment`, a cross-encoder natural-language-inference model
  scoring whether the context entails the answer.
- **api**: `llm_judge`, a strong model (default Claude) that judges faithfulness
  and serves as the reference detector.

The NLI and LLM detectors are provider-decoupled. Each takes an injected scoring
function, so the benchmark runs offline with deterministic fakes and the heavy
dependencies stay optional.

## 5. Metrics

AUROC is the primary metric because it is threshold-free and reflects a
detector's intrinsic ability to separate the two classes. We also report F1 at
each detector's F1-optimal threshold as an operating-point view. To answer the
central question we compute agreement between a cheap detector and the judge on
the binary hallucination decision, using both raw agreement rate and Cohen's
kappa, which corrects for chance agreement.

## 6. Results

Leaderboard on the bundled sample (ranked by AUROC):

| Detector | Cost | AUROC | F1 | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| novel_token | free | 0.912 | 0.850 | 0.773 | 0.944 | 0.833 |
| lexical_overlap | free | 0.887 | 0.829 | 0.739 | 0.944 | 0.806 |
| tfidf_cosine | free | 0.809 | 0.800 | 0.727 | 0.889 | 0.778 |
| majority_constant | free | 0.500 | 0.667 | 0.500 | 1.000 | 0.500 |

Detection recall by error type:

| Error type | lexical_overlap | novel_token | tfidf_cosine |
| --- | --- | --- | --- |
| contradiction | 1.00 | 1.00 | 0.75 |
| entity_swap | 0.80 | 0.80 | 1.00 |
| numeric | 1.00 | 1.00 | 0.80 |
| unsupported_addition | 1.00 | 1.00 | 1.00 |

(`majority_constant` is omitted from the breakdown discussion because it flags
every answer as hallucinated, so its per-type recall is trivially 1.00 while its
AUROC is 0.50.)

## 7. Analysis

Two findings stand out.

First, the free detectors are already strong on easy cases. `novel_token` and
`lexical_overlap` both reach AUROC near 0.9 on the sample, far above the 0.5
floor. For clear hallucinations that introduce new vocabulary, token-level
signals are enough.

Second, the failures are concentrated and interpretable. Lexical detectors lose
recall on `entity_swap`, where only one content word changes and overlap stays
high, so a wrong entity slips through. TF-IDF cosine weakens on `contradiction`
and `numeric`, where the distinctive terms still match the context even though
the claim is false. These are exactly the cases a semantic model (NLI) or an LLM
judge should catch, which is the argument for paying for a higher tier.

The cost/accuracy frontier ties this together. Among the free detectors only
`novel_token` is Pareto-optimal, since the others cost the same and detect less.
Adding the NLI model populates the mid-cost tier, and the LLM judge sits at the
top of the frontier: most expensive, highest quality. The research payoff is the
shape of that staircase, which quantifies how much AUROC each step up the cost
ladder actually buys.

## 8. Limitations

The dataset is small and hand-written, which keeps it reproducible but limits
statistical power and domain coverage. The error types are a coarse taxonomy.
The live NLI and LLM paths are not exercised in continuous integration, since
they require a model download or API access, so the committed results cover the
free tier. Absolute cost is represented by relative per-tier scores rather than
measured dollars.

## 9. Future Work

Natural next steps are a larger and harder dataset with annotated unsupported
spans, measured dollar and latency costs for the model and API tiers, more
baselines in the mid-cost tier (embedding similarity, smaller NLI models), and a
reported agreement study between the cheap detectors and the judge across the
full dataset.

## 10. Reproducibility

The free-tier results, metrics, and frontier are pure-stdlib and deterministic.
Running `python scripts/generate_report.py` regenerates `RESULTS.md`, and the
test suite (`pytest`) covers the metrics, baselines, breakdown, frontier,
agreement, and report generation offline.
