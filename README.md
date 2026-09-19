# Korean Learner Correction User Benchmark

This repo contains the code for a benchmark tool.

Goal of this benchmark is to score LLMs on zero-shot corrections for korean language learners.

## Structure

The benchmark uses an OpenAI compatible endpoint to score the [KoLLA v2](data/README.md) dataset.
By default every sentence is sent as the user message, after this system message
(the `simple` system prompt of [scripts/prompts.json](scripts/prompts.json)):

```
Correct the Korean sentence. Reply with the corrected sentence only.
```

`--prompt` replaces the user message with a template; the sentence is substituted for
`{sentence}`, which it must contain. Once `--prompt` is given, no system message is sent unless
`--system` sets one; `--system` alone replaces the default system message and keeps the bare
sentence as user message. For example, the `extended` prompt as a single user message:

```bash
cargo run --release -- --model google/gemini-2.5-flash --prompt "Correct the following korean sentence. Only correct actual errors. Reply with only the corrected sentence. {sentence}"
```

The benchmark as it stood before the prompt and iteration experiments were merged in is tagged
`benchmark-v1`.

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

The aggregation script recursively scans `results/`, reads `results/cost.csv` and
`scripts/model-info.csv`, and writes `results/metrics.csv` and `results/run-details.csv`.
Both files are the inputs of the benchmark presentation page. Display name, company, and the
weights of local models cannot be read from a run, add them to `scripts/model-info.csv` for
every new model; models without an entry are written with those columns left empty.

## Remote runs on vast.ai

Self-hosted models can be benchmarked on a rented [vast.ai](https://vast.ai) GPU. One
instance is rented per batch. It serves every model of a list with
[vLLM](https://docs.vllm.ai) one after the other, benchmarks it, uploads the run to an
S3-compatible bucket as soon as it is done, and destroys itself when the list is done or the
deadline passes. The local machine can go offline once `launch` returns.

Put the credentials into the environment or `.env` at the repository root:

```
VAST_API_KEY=...
S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
S3_BUCKET=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_REGION=auto            # optional
HF_TOKEN=...              # optional, for gated models
```

The S3 credentials and `HF_TOKEN` are handed to the instance, so prefer a token scoped to
the bucket. The vast.ai API key stays local; the instance destroys itself with its own
per-instance key.

Models are listed like [scripts/vllm.txt](scripts/vllm.txt): a Hugging Face repo per line in
a format vLLM serves natively (BF16, FP8, INT8, AWQ, GPTQ), optionally followed by extra
`vllm serve` flags. `--max-model-len 16384`, `--max-num-seqs 128` and `--language-model-only` (the benchmark only
sends text, so vision encoders are neither loaded nor profiled) are added unless a line sets
them, and
`--tensor-parallel-size` when the instance has more than one GPU.

Reasoning models need the matching `--reasoning-parser` (e.g. `qwen3`, `gemma4`), or vLLM
returns the think block as part of the answer. The runner asks every model one sentence
before the benchmark and fails it when the answer contains reasoning markers such as
`<think>`; a finished run whose answers still contain them is moved to
`logs/<model>.rejected.json` instead of `results/`.

```bash
uv run scripts/vast/vast_bench.py launch --models scripts/vllm.txt --gpu 'gpu_ram>=80 num_gpus=1' --max-dph 3 -- --limit 0 --concurrency 32
uv run scripts/vast/vast_bench.py status
uv run scripts/vast/vast_bench.py pull --logs
uv run scripts/vast/vast_bench.py destroy
```

| Command | Description |
|---|---|
| `launch` | Rent the cheapest offer matching `--gpu` (vast.ai filter syntax, `gpu_ram` in GB), `--disk` (200 GB) and `--max-dph`, then start the batch. Offers that do not boot within `--boot-timeout` (30m) are destroyed and the next is tried. `--deadline` (12h) destroys the instance no matter what. `--pip PACKAGE` (repeatable) installs extra Python packages on the instance before serving, e.g. `cohere-melody` for Cohere's reasoning parser. `--prompts FILE` runs every prompt of a prompt list against each model, see below. Everything after `--` goes to the benchmark. |
| `status [BATCH]` | Phase, GPU, cost so far, last heartbeat, and the state and F0.5 of every model. |
| `pull [BATCH]` | Download the runs into `results/<batch>/` (`multiprompt-results/<batch>/` for a batch with `--prompts`); `--logs` also fetches the vLLM and benchmark logs. |
| `destroy [BATCH]` | Destroy the batch's instance by hand. |

`BATCH` defaults to the latest batch in the bucket. The benchmark is built on the instance
from the working tree, uncommitted changes to tracked files included. A model that vLLM
cannot load or that fails the benchmark is marked failed and the batch continues. Every run
gets a `<model>.vast.json` next to it with GPU, vLLM version, timings, and the GPU cost of
that run; `collect_metrics.py` reports these runs with provider `Vast.ai` and takes their
cost from it unless `results/cost.csv` has one.

The [multi-prompt experiment](#multi-prompt-experiment) runs on vast.ai with `--prompts`. Each
model is served once and benchmarked once per prompt before the next model loads, so a prompt
list costs one model download and startup per model, not one per prompt:

```bash
uv run scripts/vast/vast_bench.py launch --models scripts/vllm-1gpu.txt --prompts scripts/prompts.json --gpu 'gpu_ram>=80 num_gpus=1' --max-dph 3 --deadline 24h -- --limit 0 --concurrency 32
```

The prompt list is checked before anything is rented, and `--prompt`/`--system` cannot be passed
after `--` with it. Runs are named like those of `run-models.sh`: `<model>/p1.json` to `pN.json`,
each with its own `.vast.json` that carries an equal share of the model's startup. A prompt that
fails is reported and the remaining prompts still run; after a restart only the prompts not
finished yet run again. `status` lists the F0.5 of every prompt below its model, and `pull` puts
such a batch into `multiprompt-results/<batch>/` unless `--results-dir` says otherwise, where
`collect_prompt_metrics.py` picks it up. Allow for the longer batch in `--deadline`: every model
now runs the whole corpus once per prompt.

Both this script and the multi-prompt collector describe each run's model next to its id.
`provider` comes from the run's endpoint (`OpenRouter`, `Local` for localhost, `Vast.ai`, or
the endpoint's host).
`company`, `params`, `quantization` and `file_bytes` come from
[scripts/model-info.csv](scripts/model-info.csv), which is maintained by hand. A model missing
from that file gets empty cells. For LM Studio models, `lms ls --json` lists the parameters,
quantization and file size. OpenRouter doesn't expose how a hosted model is served, so those
cells stay empty.

Full command usage:

```bash
Usage: kolla-benchmark [OPTIONS]

Options:
      --api-key <API_KEY>          API Key to send along with requests [env: API_KEY=]
  -m, --model <MODEL>              Model to evaluate
  -u, --url <URL>                  Base URL for OpenAI compatible request [default: https://openrouter.ai/api/v1]
      --system <SYSTEM>            System prompt sent before the user message [default: the original benchmark instruction, unless --prompt or --iterate is given; then no system message is sent]
      --prompt <PROMPT>            User prompt template; {sentence} is replaced by the challenge sentence [default: the bare sentence, or the extended prompt with --iterate]
  -d, --data <DATA>                KoLLA M2 annotations to evaluate against [default: data/KoLLA_multi-refs.m2]
  -l, --limit <LIMIT>              Sentences to evaluate; 0 runs the whole corpus (that costs real money) [default: 25]
  -c, --concurrency <CONCURRENCY>  Requests in flight at the same time [default: 10]
      --iterate                    Send every answer back as the sentence to correct until the model returns it unchanged
      --max-iterations <N>         Requests per sentence before an iterated sentence stops without settling [default: 10]
  -o, --output <OUTPUT>            Where to write the run; defaults to results/<model>-<timestamp>.json, or to iterate-results/ with --iterate
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
| `korean` | The same instructions in Korean, ending in `문장: {sentence}` / `교정문:`. |

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


## Iterated correction experiment

`--iterate` tests whether a model settles on its own correction. Each sentence is first sent as
usual. The answer is then sent back in a fresh single-turn request, in place of the sentence, until
an answer tokenizes the same as the text it was sent: the model considers it correct. A model that
returns the original unchanged therefore takes one request. `--max-iterations` (default 10) caps
the requests per sentence; a sentence that reaches it is scored on its last answer and counted as
not converged. The default prompt of this experiment is `extended`, as a single user message;
`--prompt` and `--system` still override it.

```powershell
scripts\run-models.ps1 --models scripts\local.txt --results-dir iterate-results --iterate --limit 0
```

Every answer is recorded in the result's `rounds`, but only the last one is scored. Each request is
retried like a normal one; a request that still fails fails the whole sentence, whose failure
entry keeps the rounds received before it. A sentence keeps its concurrency slot for its whole
chain. The run's `summary.requests` holds how many sentences converged and the requests per scored
sentence: total, p25, median, mean, p75, p90 and max, percentiles interpolated linearly.

Its runs are collected with their own script, which writes `iterate-metrics.csv`, one row per run:

```bash
uv run --no-project python ./scripts/collect_iterate_metrics.py --results-dir iterate-results
```

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
