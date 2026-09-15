#!/usr/bin/env bash
# Runs on the rented vast.ai instance: builds the benchmark, runs every model of the batch,
# uploads what it produced and destroys the instance.
#
# The onstart command vast_bench.py sets downloads this file, runner.py and common.py into
# /workspace/kolla and starts it. vast.ai runs it again when the container restarts; the
# runner then continues where the batch stopped.
set -uo pipefail

work=/workspace/kolla
cd "$work"
mkdir -p logs
exec > >(tee -a logs/runner.log) 2>&1
echo "=== onstart $(date -u +%FT%TZ), batch $KOLLA_BATCH ==="

finish() {
    python3 runner.py upload-log || true
    python3 runner.py destroy
    exit "$1"
}

fail() {
    echo "$1"
    python3 runner.py phase failed || true
    finish 1
}

# Destroy the instance at the deadline, whatever the rest of this script is doing.
(
    remaining=$((KOLLA_DEADLINE_AT - $(date +%s)))
    [ "$remaining" -gt 0 ] && sleep "$remaining"
    echo "deadline reached"
    finish 1
) &

# Custom variables are not visible in SSH sessions otherwise.
env | grep -E '^(KOLLA_|S3_|HF_TOKEN=)' >>/etc/environment
grep -q ' vast.local$' /etc/hosts || echo "127.0.0.1 vast.local" >>/etc/hosts

# The image's Python may be marked externally managed, hence the fallbacks.
python3 -m pip install --quiet boto3 ||
    python3 -m pip install --quiet --break-system-packages boto3 ||
    uv pip install --system --break-system-packages --quiet boto3 ||
    fail "could not install boto3"
python3 runner.py phase booting || fail "could not write the batch status"

python3 runner.py phase building || true
if [ ! -x source/target/release/kolla-benchmark ]; then
    python3 runner.py fetch || fail "could not fetch the benchmark source"
    if ! command -v cc >/dev/null || ! command -v curl >/dev/null; then
        apt-get update -qq &&
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends build-essential curl ca-certificates ||
            fail "could not install build tools"
    fi
    if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
        curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal || fail "could not install rust"
    fi
    (cd source && "$HOME/.cargo/bin/cargo" build --release) || fail "could not build the benchmark"
fi

python3 runner.py run || echo "runner exited with $?"
finish 0
