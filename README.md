# Korean Learner Correction User Benchmark

This repo contains the code for a benchmark tool.

Goal of this benchmark is to score LLMs on zero-shot corrections for korean language learners.

## Structure

The benchmark uses an OpenAI compatible endpoint to score the [KoLLA v2](data/README.md) dataset.
The sentence is substituted for `{sentence}` in the following prompt, which is sent as a single user message:

```
아래 한국어 문장에서 틀린 부분만 수정하고, 교정 문장 한 줄만 출력하라.
단, 문장이 이미 문법적으로 올바르면 원문을 그대로 출력하라.
문장: {sentence}
교정문:
```

A different template can be passed with `--prompt`; it must contain `{sentence}`.
`--system` adds a system message before the user message, sent verbatim; without it no system message is sent.

```bash
cargo run --release -- --model google/gemini-2.5-flash --system "Correct the Korean sentence. Reply with the corrected sentence only." --prompt "{sentence}"
```

KoLLA v2 contains two annotations and the result from the API is scored against both.
The better score is chosen for that particular sentence.

In its default configuration, the will attempt 3 retries for most expectedly retriable errors: 500 type errors and 429 and unparseable response errors specifically.

## Running

From PowerShell at the repository root:

```powershell
cargo run --release -- --model google/gemini-2.5-flash --limit 50
```

From Bash:

```bash
cargo run --release -- --model google/gemini-2.5-flash --limit 50
```

Model execution and scoring are separate. The Python M2 graph algorithm port is the active Rust scorer.

| Command | Description |
|---|---|
| `--model M` | Ask the model, score the answers, and write the run to `results/`. |
| `--rescore results/run.json` | Rescore an existing run without API calls; print the report only. |
| `--rescore results/run.json --output rescore/run.json` | Rescore an existing run and write a new JSON file. |
| `--baseline` | Score the corpus against itself without API calls. |

`--limit` defaults to **25 sentences** so that a mistyped command cannot cost much.
Use `--limit 0` to run all 1418 sentences.

Each run is written as one JSON file containing the provenance (model, endpoint, prompt,
dataset path and hash, tool version, timestamp), corpus metrics, and every sentence with
its answer and edits—enough to rescore it later without the corpus.

To aggregate all JSON result files into CSV:

```bash
uv run --no-project python ./scripts/collect_metrics.py
```

The aggregation script recursively scans `results/`, reads `results/cost.csv`, and writes
`results/metrics.csv`.

Full command usage:

```bash
Usage: kolla-benchmark [OPTIONS]

Options:
      --api-key <API_KEY>          API Key to send along with requests [env: API_KEY=]
  -m, --model <MODEL>              Model to evaluate
  -u, --url <URL>                  Base URL for OpenAI compatible request [default: https://openrouter.ai/api/v1]
      --system <SYSTEM>            System prompt sent before the user message; no system message is sent without it
      --prompt <PROMPT>            User prompt template; {sentence} is replaced by the challenge sentence [default: "아래 한국어 문장에서 틀린 부분만 수정하고, 교정 문장 한 줄만 출력하라.\n단, 문장이 이미 문법적으로 올바르면 원문을 그대로 출력하라.\n문장: {sentence}\n교정문:"]
  -d, --data <DATA>                KoLLA M2 annotations to evaluate against [default: data/KoLLA_multi-refs.m2]
  -l, --limit <LIMIT>              Sentences to evaluate; 0 runs the whole corpus (that costs real money) [default: 25]
  -c, --concurrency <CONCURRENCY>  Requests in flight at the same time [default: 10]
  -o, --output <OUTPUT>            Where to write the run; defaults to results/<model>-<timestamp>.json
      --rescore <RESCORE>          Score a previously written run again instead of calling the API
      --baseline                   Score the corpus against itself instead of calling the API
  -h, --help                       Print help
  -V, --version                    Print version
```

## Concurrency

`--concurrency` requests are kept in flight for as long as sentences remain: the moment any
request finishes the next one starts, even when an earlier sentence is still waiting on a slow
answer. Answers are scored only once all requests are done, so scoring never delays the network,
and are written in corpus order. A request waiting out a retry backoff keeps its slot, so an
endpoint that throttles sees fewer requests; the progress bar counts those as backing off.

For LM Studio, set `--concurrency` to the number of parallel requests the loaded model is
configured for. Requests beyond that wait in LM Studio's queue, and that waiting counts towards
the 120 second request timeout.

## Multi-prompt experiment

Separate from the primary benchmark, every model can be run against every prompt of
[scripts/prompts.json](scripts/prompts.json): four prompt styles, each as a single user message
and as a system prompt followed by the bare sentence.

| Name | Prompt |
|---|---|
| `simple` | Correct the Korean sentence. Reply with the corrected sentence only. |
| `extended` | `simple`, but only correct actual errors. |
| `long` | Multi-line instructions about learner errors, meaning preservation and style. |
| `korean` | The Korean default prompt. |

```powershell
scripts\run-models.ps1 --models scripts\local.txt --prompts scripts\prompts.json --limit 0
```

Its runs are collected with their own script:

```bash
uv run --no-project python ./scripts/collect_prompt_metrics.py --results-dir multiprompt-results
```

It writes `prompt-metrics.csv`, one row per run, and `prompt-matrix.csv`, the F0.5 of every model
(rows) for every prompt variant (columns; the latest run wins when a model ran a variant twice),
into the results directory. Every run is labelled along two axes. `prompt_name` is the `name` of
the entry in `scripts/prompts.json` whose system prompt and user template the run sent (`custom`
when none matches), so renaming a prompt there relabels old runs too. `prompt_type` is `user` when
the prompt went out as a single user message and `system+user` when a system prompt came first.
There is no cost column: the cost file holds one total per model, which cannot be split by prompt.


## Scoring

The rust implementation is built to mirror [ayaka14732/m2scorer](https://github.com/ayaka14732/m2scorer).
Results can be verified for scoring parity using:

```bash
git clone --branch py3 https://github.com/ayaka14732/m2scorer.git ..\m2scorer
uv run --no-project python .\scripts\verify_m2.py .\results\run.json --reference-dir ..\m2scorer\scripts
```

## Baselines

`--baseline` scores the corpus against itself. It is also the scorer's self-test:

```
1418 sentences, 2828 annotations, 3649 gold edits

answer unchanged   P 0.0000  R 0.0000  F0.5 0.0000   (tp 0, fp 0, fn 1380)
human annotator    P 0.9992  R 1.0000  F0.5 0.9993   (tp 3649, fp 3, fn 0)

tokenizer          3 of 2828 annotations tokenize differently than the corpus
maxmatch           0 of 2828 annotations are not fully expressible as edits on a
                   minimum distance alignment, and can never be scored in full
```

## Results

The tool was ran against openrouter for the models outlined in [scripts/models.txt](scripts/models.txt) and against LM-Studio locally for [scripts/local.txt](scripts/local.txt) (often highly quantized versions).

For the open router models, the top results by F0.5 are:

| Model | Scored | Failures | Precision | Recall | F0.5 |
|---|---:|---:|---:|---:|---:|
| x-ai/grok-4.6 | 1,417 | 1 | 0.6521 | 0.7450 | **0.6688** |
| deepseek/deepseek-v4.1-flash | 1,412 | 6 | 0.6267 | 0.7222 | 0.6437 |
| moonshotai/kimi-k3 | 1,417 | 1 | 0.6210 | 0.7171 | 0.6381 |
| anthropic/claude-sonnet-5 | 1,418 | 0 | 0.6105 | 0.7253 | 0.6305 |
| anthropic/claude-opus-5 | 1,418 | 0 | 0.5807 | 0.7449 | 0.6075 |

For the local models,  results by F0.5 are:

| Model | Params | File bytes | Quant. | Scored | Failures | P | R | F0.5 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| meta/muse-glimmer | 28B | 18,157,122,004 | Q4_K_M | 1,152 | 3 | 0.5477 | 0.6600 | **0.5670** |
| google/gemma-4-26b-a4b | 26B-A4B | 17,990,911,801 | Q4_K_M | 1,410 | 8 | 0.5040 | 0.6607 | 0.5291 |
| google/gemma-4-e4b | 7.5B | 6,326,932,336 | Q4_K_M | 1,418 | 0 | 0.3822 | 0.5483 | 0.4069 |
| kanana-2-30b-a3b-thinking-2601 | 30B-a3B | 18,596,926,752 | Q4_K_M | 1,418 | 0 | 0.3478 | 0.4663 | 0.3664 |
| mistralai/ministral-3-3b | 3B | 4,491,998,767 | Q8_0 | 1,418 | 0 | 0.2631 | 0.3791 | 0.2803 |

The results were obtained between 10th and 12th of September 2026.

A full presentation should be available [here](https://elephant-project.net/benchmarks/benchmarks/korean-error-correction/).

Full result files can be downloaded from [Elephant-Project](https://media.elephant-project.net/benchmark/gec/results.zip).

## References

- Song, J., Lim, K., & Park, J. (2026). *Enriching the Korean learner corpus for grammatical error correction and writing assessment*. **Language Resources & Evaluation, 60**, Article 15. [https://doi.org/10.1007/s10579-025-09882-9](https://doi.org/10.1007/s10579-025-09882-9)

- Park, J. (2025). *Enriching KoLLA with Multi-reference Annotations and Rubric-Based Scoring* (KoLLA v2, Version 1) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.15287129](https://doi.org/10.5281/zenodo.15287129)

- Dahlmeier, D., & Ng, H. T. (2012). Better Evaluation for Grammatical Error Correction. In *Proceedings of NAACL-HLT 2012* (pp. 568–572). [ACL Anthology](https://aclanthology.org/N12-1067/)

- NUS NLP. *M²Scorer: MaxMatch Scorer*. [GitHub](https://github.com/nusnlp/m2scorer)

- ayaka14732. *m2scorer*, Python 3 port (`py3` branch). [GitHub](https://github.com/ayaka14732/m2scorer/tree/py3)

- *KoLLA v1.0 Korean Learner Corpus*. [Project page](https://cl.indiana.edu/~kolla/)
