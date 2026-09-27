#!/bin/bash
# Downloads the MultiBench affect datasets (pre-extracted feature pickles) from the
# Google Drive IDs listed in dataset/affect/get_data.py. Download only.
#
#   bash scripts/fetch_multibench.sh <dest_dir> [dataset ...]
#   bash scripts/fetch_multibench.sh /data/multibench                 # mosi mosei humor sarcasm
#   bash scripts/fetch_multibench.sh /data/multibench mosi sarcasm
#
# Needs `gdown` in the active environment (it is in environment.yml).

set -u

DEST=${1:?destination directory}
shift
WANT=("$@")
[ "${#WANT[@]}" -eq 0 ] && WANT=(mosi mosei humor sarcasm)

declare -A ID=(
    [mosi]=1_XdzdW8UNG1TTS6QcX10uhoS6N11OBit
    [mosei]=180l4pN6XAv8-OAYQ6OrMheFUMwtqUWbz
    [humor]=1L5slPmYyhEVtwGyM1kgcFMjeBpXLZGT0
    [sarcasm]=1EMBUmUL5B0PTncGx3L-sBElGOmjFBR_h
)
declare -A FILE=(
    [mosi]=mosi_data.pkl
    [mosei]=mosei_senti_data.pkl
    [humor]=humor.pkl
    [sarcasm]=sarcasm.pkl
)

command -v gdown >/dev/null || { echo "gdown not found in the active environment"; exit 1; }
mkdir -p "$DEST"

# Drive answers a quota or permission error with a small HTML page saved under the
# target name, so a zero exit code is not enough: check that a pickle landed.
verify() {
    local f=$1 sz kind
    [ -s "$f" ] || { echo "  missing or empty"; return 1; }
    sz=$(stat -c %s "$f")
    kind=$(file -b "$f")
    if [ "$sz" -lt 1000000 ] || printf '%s' "$kind" | grep -qiE 'html|ascii|text'; then
        echo "  rejected: $(numfmt --to=iec "$sz") of '$kind' (Drive quota page, not the dataset)"
        return 1
    fi
    echo "  ok: $(numfmt --to=iec "$sz")"
}

failed=()
for ds in "${WANT[@]}"; do
    if [ -z "${ID[$ds]:-}" ]; then
        echo "[$ds] unknown dataset; known: ${!ID[*]}"
        failed+=("$ds"); continue
    fi
    out="$DEST/${FILE[$ds]}"
    echo "--- [$ds] -> $out"
    if verify "$out" >/dev/null 2>&1; then
        echo "  already present"
        continue
    fi
    rm -f "$out"
    gdown "https://drive.google.com/uc?id=${ID[$ds]}" -O "$out"
    verify "$out" || { failed+=("$ds"); rm -f "$out"; }
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}"
    echo "Drive rate-limits public files: retry later, or download them in a browser into $DEST."
    exit 1
fi
