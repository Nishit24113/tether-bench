# Tether

**A reproducible benchmark for RAG answer faithfulness.**

Retrieval-augmented generation (RAG) systems ground an LLM's answer in retrieved
context — but the model can still *hallucinate*, asserting things the context
never supports. Catching that is critical in production. A strong LLM-judge can
do it, but running one on every answer is slow and expensive. Teams want **cheap
faithfulness detectors** they can run inline.

Tether asks a focused research question:

> **How well do low-cost, zero-API faithfulness detectors separate grounded
> answers from hallucinations — and how closely can they approximate an
> LLM judge?**

The repo provides a labeled dataset, a rigorous evaluation harness, and a
growing suite of baselines spanning the **cost/accuracy frontier** — from
free lexical heuristics to (in later versions) a Claude-based judge.

## Why this is a benchmark, not a demo

- **Reproducible by construction.** The v0 baselines and metrics are pure
  Python stdlib — no model downloads, no API keys, no GPU. `pytest` and the
  leaderboard produce identical numbers on any machine.
- **A clear task + protocol.** Binary faithfulness labels, a threshold-free
  primary metric (AUROC), and F1 at the oracle-optimal threshold as a secondary
  operating-point view.
- **A floor and a frontier.** A majority baseline pins AUROC at 0.5; every real
  detector is measured against it and tagged with a cost tier for the
  cost/accuracy analysis.

## Task

Each example is a `(question, context, answer)` triple with a gold label:

- `faithful = true` — every claim in the answer is supported by the context
- `faithful = false` — the answer asserts something the context does not

A **detector** maps an example to a faithfulness score in `[0, 1]` (higher =
more grounded). The harness converts scores to hallucination predictions via a
threshold and reports precision / recall / F1 / accuracy / AUROC. The positive
class is *hallucinated* — the thing we want to catch.

## Baselines (v0)

| Detector | Cost | Idea |
|----------|------|------|
| `majority_constant` | free | Constant score — the 0.5-AUROC floor |
| `lexical_overlap` | free | Fraction of answer content tokens present in context |
| `novel_token` | free | Penalizes answer tokens absent from context *and* question |
| `tfidf_cosine` | free | TF-IDF cosine similarity between answer and context |
| `nli_entailment` | model | A local NLI model scores whether the context entails the answer — the mid-cost tier |
| `llm_judge` | api | A strong model (default `claude-opus-5-5`) judges faithfulness — the reference the cheap detectors are measured against |

All three cost tiers (`free`, `model`, `api`) are now populated, so the
cost/accuracy frontier spans the full ladder.

## Results on the bundled sample (n=36, balanced)

```
detector           cost  auroc   f1      precision  recall  accuracy
-----------------  ----  ------  ------  ---------  ------  --------
novel_token        free  0.9120  0.8500  0.7727     0.9444  0.8333
lexical_overlap    free  0.8873  0.8293  0.7391     0.9444  0.8056
tfidf_cosine       free  0.8086  0.8000  0.7273     0.8889  0.7778
majority_constant  free  0.5000  0.6667  0.5000     1.0000  0.5000
```

Even trivial lexical detectors clear the floor by a wide margin on easy cases.
The interesting question is *where* they fail — which the error-type breakdown
below answers directly.

## Error-type breakdown

Every hallucinated example is annotated with the kind of error it contains, and
every faithful one with how it is supported. Tether reports per-category
**detection recall** — what fraction of each error kind a detector actually
catches at its F1-optimal threshold:

```
Detection recall by error type (hallucinated examples)

error_type            majority_constant  lexical_overlap  novel_token  tfidf_cosine
--------------------  -----------------  ---------------  -----------  ------------
contradiction         1.00               1.00             1.00         0.75
entity_swap           1.00               0.80             0.80         1.00
numeric               1.00               1.00             1.00         0.80
unsupported_addition  1.00               1.00             1.00         1.00
```

(`majority_constant` scores 1.00 everywhere only because it flags *everything* as
hallucinated — its 0.50 AUROC is the honest signal; recall here is a reminder
that per-category recall must be read alongside false-positive rate.)

The real finding is that the error types that evade cheap detectors are the ones
where the hallucinated answer **reuses the context's vocabulary**:

- **`entity_swap`** (e.g. "the *liver* produces insulin") slips past token-overlap
  detectors — only one content word changed, so overlap stays high.
- **`numeric`** errors (a single wrong digit) and **`contradiction`** are where
  TF-IDF weakens, since the distinctive terms still match.

Catching exactly these cases is where a semantic NLI model or an LLM judge should
earn its cost — the comparison the next milestones add.

## The LLM judge & agreement analysis

The whole point of Tether is to ask *how closely a free detector tracks an
expensive LLM judge*. The judge is the reference detector; agreement with it is
the headline metric.

The judge is provider-decoupled — `LLMJudgeDetector` takes any `judge_fn`
mapping an example to a faithfulness score, so it's trivially mockable and the
test suite runs offline. The Anthropic-backed judge is opt-in
(`pip install tether-bench[llm]`, `ANTHROPIC_API_KEY` set):

```python
from tether import (
    load_sample, LexicalOverlapDetector, LLMJudgeDetector,
    anthropic_judge, best_threshold, judge_agreement,
)

examples = load_sample()

# Reference judge (a strong model). Any callable works; this one calls Claude.
judge = LLMJudgeDetector(anthropic_judge(model="claude-opus-5-5"))
judge_scores = judge.score_all(examples)

# A cheap detector at its F1-optimal threshold
cheap = LexicalOverlapDetector()
cheap_scores = cheap.score_all(examples)
labels = [e.label for e in examples]
thr, _ = best_threshold(labels, cheap_scores)

# How closely does the cheap detector track the judge?
print(judge_agreement(cheap_scores, judge_scores, cheap_threshold=thr))
# {'agreement_rate': ..., 'cohen_kappa': ..., 'cheap_only_flags': ..., 'judge_only_flags': ..., ...}
```

`cohen_kappa` corrects for chance agreement; `judge_only_flags` counts the
hallucinations the judge caught that the cheap detector missed — exactly the gap
a paid judge is buying.

The **mid-cost tier** is the NLI detector — same decoupled design, so it's
mockable in tests and the model is an opt-in dependency
(`pip install tether-bench[nli]`):

```python
from tether import NLIEntailmentDetector, transformers_nli

# premise = context, hypothesis = answer; score = P(context entails answer)
nli = NLIEntailmentDetector(transformers_nli("facebook/bart-large-mnli"))
scores = nli.score_all(examples)
```

## Cost/accuracy frontier

The benchmark's headline artifact: each detector placed in (cost, quality) space,
with the Pareto-optimal ones marked. Cost is a relative per-tier score (free
compute = 0, local model = 0.3, paid API = 1.0) — override `COST_SCORE` with
measured costs when you have them.

```
Cost/accuracy frontier (quality = auroc; * = Pareto-optimal)

detector           cost_tier  cost  auroc  frontier
-----------------  ---------  ----  -----  --------
novel_token        free       0.00  0.912  *
lexical_overlap    free       0.00  0.887
tfidf_cosine       free       0.00  0.809
majority_constant  free       0.00  0.500
```

Among the free detectors only `novel_token` is Pareto-optimal — the rest cost
the same but detect less. Add the `llm_judge` row (cost tier `api`) and it joins
the frontier at the top: more expensive, but highest quality. The research
payoff is the shape of that staircase — *how much AUROC each step up the cost
ladder actually buys.*

```python
from tether import load_sample, default_detectors, run_benchmark, build_frontier, format_frontier

examples = load_sample()
bench = run_benchmark(default_detectors(examples), examples)
print(format_frontier(build_frontier(bench["leaderboard"], metric="auroc")))
```

## Quickstart

```bash
pip install -e ".[dev]"
pytest                          # 38 tests, all stdlib
python scripts/run_benchmark.py # prints the leaderboard above
```

Run against your own data (JSONL, one example per line):

```json
{"id": "1", "question": "...", "context": "...", "answer": "...", "faithful": true}
```

```bash
python scripts/run_benchmark.py path/to/your.jsonl
```

## Layout

```
tether/
  data.py            # Example type + JSONL / sample loaders
  metrics.py         # confusion matrix, AUROC (Mann-Whitney), threshold search
  harness.py         # run detectors -> metrics -> ranked leaderboard
  agreement.py       # agreement rate + Cohen's kappa vs the judge
  frontier.py        # cost/accuracy Pareto frontier
  baselines/         # majority, lexical, novel-token, tf-idf, nli, llm-judge
data/sample.jsonl    # bundled labeled dataset (runs offline), error-type annotated
scripts/run_benchmark.py
tests/               # 99 unit tests
```

## Roadmap

- **Larger, harder dataset** with annotated unsupported spans
- **Paper draft**: benchmark description, baseline analysis, cost/accuracy findings

_Done:_ error-type / support-type annotations with per-category recall breakdown;
NLI-entailment mid-cost baseline, LLM-judge reference detector (both
provider-decoupled) + agreement analysis (agreement rate, Cohen's kappa);
cost/accuracy frontier with Pareto marking.

## Reproducibility

v0 is deterministic and dependency-free. Later baselines will pin model
versions and seeds, and record per-example scores so every leaderboard number is
traceable to its inputs.

---

**Built by Nishit Patel** | MS Computer Science, Arizona State University
