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

Planned: `nli_entailment` (local NLI model) and `llm_judge` (Claude) — the
reference point the cheap detectors are measured against.

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
  baselines/         # majority, lexical, novel-token, tf-idf
data/sample.jsonl    # bundled labeled dataset (runs offline), error-type annotated
scripts/run_benchmark.py
tests/               # 49 unit tests
```

## Roadmap

- **LLM-judge baseline** (Claude) as the reference detector + agreement analysis
- **NLI-entailment baseline** (local model) for a mid-cost tier
- **Larger, harder dataset** with annotated unsupported spans
- **Cost/accuracy frontier** plot
- **Paper draft**: benchmark description, baseline analysis, cost/accuracy findings

_Done:_ error-type / support-type annotations and per-category recall breakdown.

## Reproducibility

v0 is deterministic and dependency-free. Later baselines will pin model
versions and seeds, and record per-example scores so every leaderboard number is
traceable to its inputs.

---

**Built by Nishit Patel** | MS Computer Science, Arizona State University
