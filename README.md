# kolla-benchmark

Benchmarks an OpenAI compatible endpoint on [KoLLA v2](data/README.md).

## Running

```bash
cargo run --release -- --model google/gemini-2.5-flash --limit 50
```

Model execution and scoring are separate, and two of the three modes cost nothing:

|                              |                                                               |
|------------------------------|---------------------------------------------------------------|
| `--model M`                  | ask the model, score the answers, write the run to `results/` |
| `--rescore results/run.json` | score the answers of an earlier run again, no API calls       |
| `--baseline`                 | score the corpus against itself, no API calls                 |

`--limit` defaults to **25 sentences** so that a mistyped command cannot cost much;
`--limit 0` runs all 1418. Other flags: `--url`, `--prompt`, `--data`, `--concurrency`,
`--output`, and `--api-key` (also read from `API_KEY`). A sentence that keeps failing is
recorded in `failures`; three consecutive retry-exhausted transient failures stop the run.
Retryable requests honor `Retry-After` when provided and otherwise use capped exponential
backoff with jitter. Retry and failure details are written to stderr with sentence progress.
Individual HTTP requests time out after 120 seconds.

Each run is written as one JSON file containing the provenance (model, endpoint, prompt,
dataset path and hash, tool version, timestamp), the corpus metrics, and every sentence
with its answer and its edits — enough to re-score it later without the corpus.

## How it is scored

1. The learner sentence goes to the model wrapped in `--prompt`, and one answer comes back.
2. The answer is tokenized like the M2 source lines: on whitespace, with sentence-final
   punctuation split off. No morphological tokenization — it would not fit the M2 offsets.
3. MaxMatch searches all decompositions of the answer into edits along a minimum distance
   alignment, merging adjacent operations into phrase-level edits, and keeps the one that
   matches the most gold edits (fewest edits among equals). It is not a diff.
4. An edit counts as correct when span and replacement match a gold edit. Error categories
   are not used.
5. Every reference is scored separately and the best F0.5 per sentence is kept.
6. TP/FP/FN of the chosen references are summed over the corpus, and P, R and F0.5 are
   computed once from those sums — never an average of per-sentence F0.5.

## Baselines

`--baseline` scores the corpus against itself. It is also the scorer's self-test:

```
1418 sentences, 2828 annotations, 3649 gold edits

answer unchanged   P 0.0000  R 0.0000  F0.5 0.0000   (tp 0, fp 0, fn 1380)
human annotator    P 0.9956  R 0.9926  F0.5 0.9950   (tp 3617, fp 16, fn 27)

tokenizer          3 of 2828 annotations tokenize differently than the corpus
maxmatch           12 of 2828 annotations are not fully expressible as edits on a
                   minimum distance alignment, and can never be scored in full
```
