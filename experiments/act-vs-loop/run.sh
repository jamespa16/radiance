#!/usr/bin/env bash
# Run every config in a directory serially, one at a time.
#
# Serial is deliberate, not incidental: CLAUDE.md records two hyper-connection arms invalidated by
# a diagnostic run started alongside them, where the symptom was an OOM-shaped early exit in the
# *other* process. Each arm gets the whole GPU.
set -uo pipefail

# uv resolves the project from the cwd, so the driver must run inside the repo no matter
# where it was invoked from -- otherwise every arm fails instantly with "Failed to spawn".
cd /home/james/dev/radiance || exit 1

DIR="$1"
LOGS="$DIR/logs"
mkdir -p "$LOGS"

for cfg in "$DIR"/*.yaml; do
    name=$(basename "$cfg" .yaml)
    log="$LOGS/$name.log"
    echo "[driver] === $name ==="
    start=$SECONDS
    uv run radiance-train --config "$cfg" >"$log" 2>&1
    rc=$?
    elapsed=$((SECONDS - start))

    # An OOM-terminated arm still exits 0 and still prints evals, so it looks like a valid short
    # run rather than a failed one. CLAUDE.md calls this out specifically -- check for it.
    if grep -q "ending run early" "$log"; then
        echo "[driver] $name INVALID: 'ending run early' (OOM) after ${elapsed}s"
        continue
    fi
    if [ $rc -ne 0 ]; then
        echo "[driver] $name FAILED rc=$rc after ${elapsed}s"
        tail -5 "$log" | sed 's/^/[driver]   /'
        continue
    fi
    final=$(grep "val/loss" "$log" | tail -1)
    echo "[driver] $name done in ${elapsed}s :: ${final:-NO EVAL}"
done
echo "[driver] ALL DONE"
