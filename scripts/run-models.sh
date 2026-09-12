#!/usr/bin/env bash
# Benchmark a list of models, one after the other.
#
#   API_KEY=... scripts/run-models.sh --limit 100
#   API_KEY=... scripts/run-models.sh --models other-list.txt --limit 0
#
# Everything but --models is passed on to the benchmark unchanged. One model failing does
# not stop the others. Runs land in results/<timestamp>/<model>.json.
set -uo pipefail

cd "$(dirname "$0")/.."

# Read out --models wherever it stands, pass everything else on to the benchmark.
models_file="scripts/models.txt"
forward=()
while [ $# -gt 0 ]; do
    case "$1" in
    --models)
        models_file="${2:?--models needs a file}"
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

cargo build --release || exit 1
binary="target/release/kolla-benchmark"
[ -f "$binary.exe" ] && binary="$binary.exe"

batch="results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$batch"

# The file a model writes to, its name stripped of anything a path would not like.
result_of() { printf '%s/%s.json' "$batch" "$(printf '%s' "$1" | tr -c 'a-zA-Z0-9.-' '_')"; }

for model in "${models[@]}"; do
    echo
    echo "=== $model ==="
    "$binary" --model "$model" --output "$(result_of "$model")" "${forward[@]}" ||
        echo "  $model failed, continuing" >&2
done

# Pull the metrics back out of the written runs — pretty printed JSON, one key per line.
value() { sed -n "s/.*\"$2\": \([0-9.eE+-]*\).*/\1/p" "$1" | head -1; }

echo
printf '%-45s %9s %9s %9s\n' "model" "precision" "recall" "F0.5"
for model in "${models[@]}"; do
    result=$(result_of "$model")
    if [ -f "$result" ]; then
        printf '%-45s %9.4f %9.4f %9.4f\n' "$model" \
            "$(value "$result" precision)" "$(value "$result" recall)" "$(value "$result" f05)"
    else
        printf '%-45s %9s\n' "$model" "no result"
    fi
done

echo
echo "runs written to $batch"
