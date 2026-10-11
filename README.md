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

## Results on the bundled sample (n=24, balanced)

```
detector           cost  auroc   f1      precision  recall  accuracy
-----------------  ----  ------  ------  ---------  ------  --------
lexical_overlap    free  0.8958  0.8462  0.7857     0.9167  0.8333
novel_token        free  0.8958  0.8462  0.7857     0.9167  0.8333
tfidf_cosine       free  0.8611  0.8462  0.7857     0.9167  0.8333
majority_constant  free  0.5000  0.6667  0.5000     1.0000  0.5000
```

Even trivial lexical detectors clear the floor by a wide margin on easy cases —
the open question Tether is built to study is how far that holds as cases get
harder (hallucinations that reuse context vocabulary, faithful paraphrases that
share few tokens), and where cheap methods fall short of an LLM judge.

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
data/sample.jsonl    # bundled labeled dataset (runs offline)
scripts/run_benchmark.py
tests/               # 38 unit tests
```

## Roadmap

- **LLM-judge baseline** (Claude) as the reference detector + agreement analysis
- **NLI-entailment baseline** (local model) for a mid-cost tier
- **Larger, harder dataset** with annotated unsupported spans
- **Cost/accuracy frontier** plot and a per-difficulty breakdown
- **Paper draft**: benchmark description, baseline analysis, cost/accuracy findings

## Reproducibility

v0 is deterministic and dependency-free. Later baselines will pin model
versions and seeds, and record per-example scores so every leaderboard number is
traceable to its inputs.

---

**Built by Nishit Patel** | MS Computer Science, Arizona State University
