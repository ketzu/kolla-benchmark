#!/usr/bin/env bash
# Benchmark a list of models, one after the other, optionally against a list of prompts.
#
#   API_KEY=... scripts/run-models.sh --limit 100
#   API_KEY=... scripts/run-models.sh --models other-list.txt --limit 0
#   API_KEY=... scripts/run-models.sh --prompts scripts/prompts.json --limit 100
#
# Everything but --models and --prompts is passed on to the benchmark unchanged. One run failing
# does not stop the others. Runs land in results/<timestamp>/<model>.json.
#
# With --prompts every model runs against every prompt of a JSON file: an array of objects with a
# "user" template containing {sentence}, an optional "system" prompt and an optional "name". Prompts
# are numbered from 1 in file order, and runs land in results/<timestamp>/<model>/p<number>.json.
# This needs jq.
set -uo pipefail

cd "$(dirname "$0")/.."

# Read out --models and --prompts wherever they stand, pass everything else on to the benchmark.
models_file="scripts/models.txt"
prompts_file=""
forward=()
while [ $# -gt 0 ]; do
    case "$1" in
    --models)
        models_file="${2:?--models needs a file}"
        shift 2
        ;;
    --prompts)
        prompts_file="${2:?--prompts needs a file}"
        shift 2
        ;;
    *)
        forward+=("$1")
        shift
        ;;
    esac
done

if [ -z "${API_KEY:-}" ]; then
    echo "API_KEY is not set" >&2
    exit 1
fi

models=()
while IFS= read -r line; do
    line=$(printf '%s' "${line%%#*}" | tr -d '\r' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    [ -n "$line" ] && models+=("$line")
done <"$models_file"

if [ ${#models[@]} -eq 0 ]; then
    echo "no models listed in $models_file" >&2
    exit 1
fi

# Checked up front, so that a broken prompt does not surface only after hours of other runs.
prompt_count=0
if [ -n "$prompts_file" ]; then
    if ! command -v jq >/dev/null; then
        echo "--prompts needs jq" >&2
        exit 1
    fi
    for arg in "${forward[@]}"; do
        case "$arg" in
        --prompt | --system)
            echo "--prompt and --system cannot be combined with --prompts" >&2
            exit 1
            ;;
        esac
    done
    if ! jq -e 'type == "array" and length > 0 and all(.[];
            type == "object"
            and (.user | type) == "string" and (.user | contains("{sentence}"))
            and ((has("system") | not) or (.system | type) == "string")
            and ((has("name") | not) or (.name | type) == "string"))' \
        "$prompts_file" >/dev/null; then
        echo "$prompts_file must be a non-empty array of {\"name\"?: ..., \"user\": ..., \"system\"?: ...} with {sentence} in every user template" >&2
        exit 1
    fi
    prompt_count=$(jq 'length' "$prompts_file" | tr -d '\r')
fi

# Raw prompt text, byte for byte: a native Windows jq writes CRLF unless told --binary, which
# older jq versions elsewhere do not know.
raw=(jq -j)
jq -b -n 'empty' >/dev/null 2>&1 && raw+=(-b)

# Sets flags to the arguments that send prompt number $1. The appended dot keeps trailing
# newlines, which $( ) would strip.
prompt_flags() {
    local index=$(($1 - 1)) text
    text=$("${raw[@]}" --argjson n "$index" '.[$n].user + "."' "$prompts_file")
    flags=(--prompt "${text%.}")
    if jq -e --argjson n "$index" '.[$n] | has("system")' "$prompts_file" >/dev/null; then
        text=$("${raw[@]}" --argjson n "$index" '.[$n].system + "."' "$prompts_file")
        flags=(--system "${text%.}" "${flags[@]}")
    fi
}

cargo build --release || exit 1
binary="target/release/kolla-benchmark"
[ -f "$binary.exe" ] && binary="$binary.exe"

batch="results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$batch"

# Every run as a model and a prompt number; prompt number 0 runs with the flags passed on.
run_models=()
run_prompts=()
for model in "${models[@]}"; do
    if [ "$prompt_count" -eq 0 ]; then
        run_models+=("$model")
        run_prompts+=(0)
    else
        for ((n = 1; n <= prompt_count; n++)); do
            run_models+=("$model")
            run_prompts+=("$n")
        done
    fi
done

# The file a run writes to, the model name stripped of anything a path would not like.
result_of() {
    local model
    model=$(printf '%s' "$1" | tr -c 'a-zA-Z0-9.-' '_')
    if [ "$2" -eq 0 ]; then
        printf '%s/%s.json' "$batch" "$model"
    else
        printf '%s/%s/p%s.json' "$batch" "$model" "$2"
    fi
}

label_of() {
    if [ "$2" -eq 0 ]; then printf '%s' "$1"; else printf '%s, %s' "$1" "$(prompt_of "$2")"; fi
}

# Prompt number $1 with its name and message layout, e.g. "p3 korean/user".
prompt_of() {
    jq -j --argjson n "$(($1 - 1))" \
        '.[$n] | "p\($n + 1) \(.name // "unnamed")/\(if has("system") then "system+user" else "user" end)"' \
        "$prompts_file" | tr -d '\r'
}

for i in "${!run_models[@]}"; do
    model=${run_models[i]}
    prompt=${run_prompts[i]}
    label=$(label_of "$model" "$prompt")
    flags=()
    [ "$prompt" -gt 0 ] && prompt_flags "$prompt"
    echo
    echo "=== $label ==="
    "$binary" --model "$model" --output "$(result_of "$model" "$prompt")" "${flags[@]}" "${forward[@]}" ||
        echo "  $label failed, continuing" >&2
done

# Pull the metrics back out of the written runs — pretty printed JSON, one key per line.
value() { sed -n "s/.*\"$2\": \([0-9.eE+-]*\).*/\1/p" "$1" | head -1; }

# The leading columns of a summary line; the prompt column only for a prompt matrix.
columns() {
    if [ -n "$prompts_file" ]; then
        printf '%-45s %-24s ' "$1" "$2"
    else
        printf '%-45s ' "$1"
    fi
}

echo
columns "model" "prompt"
printf '%9s %9s %9s\n' "precision" "recall" "F0.5"
for i in "${!run_models[@]}"; do
    model=${run_models[i]}
    prompt=${run_prompts[i]}
    result=$(result_of "$model" "$prompt")
    if [ "$prompt" -gt 0 ]; then columns "$model" "$(prompt_of "$prompt")"; else columns "$model" ""; fi
    if [ -f "$result" ]; then
        printf '%9.4f %9.4f %9.4f\n' \
            "$(value "$result" precision)" "$(value "$result" recall)" "$(value "$result" f05)"
    else
        printf '%9s\n' "no result"
    fi
done

echo
echo "runs written to $batch"
