#!/bin/bash
#SBATCH --job-name=cell
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

# Launches one or more cells of the study on one dataset, with the seeds of each
# cell running concurrently on the same GPU. Works under `sbatch` and as a plain
# `bash` script. See the README for the full list of experiments.
#
#   bash scripts/run_cell.sh <dataset> <family> <a> <b> <c> "<seeds>" [arm]
#
#   family=womm : a=alignment (cosine|mse)  b=regularizer  c=lambda
#   family=mmsd : a=objective (dino|byol)   b=augmentation or '-'  c unused
#   arm         : Trifeatures only, biased | unbiased
#
# Several cells can share one job: set CELLS to `family:a:b:c` entries (or the
# groups below) and pass '-' for the positional cell. Cells run one after another.
#
#   CELLS="@negsamples @gauss" sbatch scripts/run_cell.sh mosi - - - - "42 1234 2026 1 7"
#
# Environment variables:
#   CONDA_ENV    conda env to activate (default: multimodal); '' uses the current python
#   CONDA_BASE   conda installation, when `conda` is not on PATH inside the job
#   REPO_DIR     repository root (default: this script's parent, or the sbatch submit dir)
#   HF_HOME      cache of the frozen pre-trained encoders (MM-IMDb, Hateful Memes)
#   MAX_PAR      seeds run at once (default: 5)
#   WORKERS      data loader workers per seed (default: 6 image-text, 4 otherwise)
#   EPOCHS       pre-training epochs (default: 100)
#   PROBE_EVERY  linear probe period in epochs (default: every epoch). The reported
#                numbers come from the probe of the final weights, which always runs.
#   HYDRA_EXTRA  extra Hydra overrides, space separated
#   RESUME=1     continue each run from its last checkpoint
#   DRY_RUN=1    print the commands without running them
#
# Learning rate, weight decay and batch size come from each dataset's train config;
# change them through HYDRA_EXTRA only for an ablation, with a `+group_suffix=` so
# its runs are named apart from the reference ones.

set -u

DATASET=${1:?dataset}
FAMILY=${2:?family (womm|mmsd)}
A=${3:?womm: alignment | mmsd: objective}
B=${4:-'-'}
C=${5:-'-'}
SEEDS=${6:?seeds, e.g. "42 1234 2026 1 7"}
ARM=${7:-'-'}

CELLS=${CELLS:-"$FAMILY:$A:$B:$C"}

expand_group() {
    case "$1" in
        @negsamples)   echo "womm:cosine:neg-samples:1.0 womm:cosine:neg-samples-dcl:1.0 \
                             womm:cosine:neg-samples-cross:1.0 womm:cosine:neg-samples-dcl-cross:1.0" ;;
        @negsamples-a) echo "womm:cosine:neg-samples:1.0 womm:cosine:neg-samples-dcl:1.0" ;;
        @negsamples-b) echo "womm:cosine:neg-samples-cross:1.0 womm:cosine:neg-samples-dcl-cross:1.0" ;;
        @gauss)        echo "womm:mse:sigreg:0.05 womm:mse:sigreg-permask:0.05 womm:mse:visreg:9.0" ;;
        @align)        echo "womm:mse:neg-samples:1.0 womm:mse:neg-samples-dcl:1.0" ;;
        @composite)    echo "womm:mse:mix-sigreg-neg-samples:1.0 womm:mse:split-sigreg-neg-samples:1.0" ;;
        @distill)      echo "mmsd:dino:-:- mmsd:byol:-:-" ;;
        @*)            echo "[job] unknown group '$1'" >&2; exit 1 ;;
        *)             echo "$1" ;;
    esac
}
_expanded=""
for _c in $CELLS; do _expanded="$_expanded $(expand_group "$_c")" || exit 1; done
CELLS=$(echo $_expanded)

# Each dataset names its entry point, its data group and whether Hydra needs the
# append form for `model` (MultiBench lists `model` in its defaults, the rest do not).
case "$DATASET" in
    trifeatures)
        MAIN=main_trifeatures.py; DATA_OV=(+data=trifeatures); MODEL_PLUS="+" ;;
    mosi|humor|sarcasm|visionandtouch|visionandtouch-bin|mimic)
        MAIN=main_multibench.py; DATA_OV=(data.data_module.dataset=$DATASET); MODEL_PLUS="" ;;
    crema_d_features)
        MAIN=main_crema_d_features.py; DATA_OV=(+data=crema_d_features); MODEL_PLUS="+" ;;
    sen1floods11)
        MAIN=main_sen1floods11.py; DATA_OV=(+data=sen1floods11); MODEL_PLUS="+" ;;
    so2sat)
        MAIN=main_so2sat.py; DATA_OV=(+data=so2sat); MODEL_PLUS="+" ;;
    mmimdb)
        MAIN=main_mmimdb.py; DATA_OV=(+data=mmimdb); MODEL_PLUS="+" ;;
    hateful_memes)
        MAIN=main_hateful_memes.py; DATA_OV=(+data=hateful_memes); MODEL_PLUS="+" ;;
    *) echo "[job] unknown dataset '$DATASET'"; exit 1 ;;
esac

if [ "$DATASET" = "trifeatures" ]; then
    case "$ARM" in
        biased)   DATA_OV+=(+data.data_module.biased=true) ;;
        unbiased) DATA_OV+=(+data.data_module.biased=false) ;;
        *) echo "[job] trifeatures needs an arm: biased | unbiased"; exit 1 ;;
    esac
elif [ "$ARM" != "-" ]; then
    echo "[job] '$DATASET' has no arms, got '$ARM'"; exit 1
fi

MAX_PAR=${MAX_PAR:-5}
case "$DATASET" in
    mmimdb|hateful_memes) WORKERS=${WORKERS:-6} ;;
    *)                    WORKERS=${WORKERS:-4} ;;
esac
EPOCHS=${EPOCHS:-100}
RESUME=${RESUME:-0}
PROTOS=${PROTOS:-2048}

HYDRA_EXTRA=${HYDRA_EXTRA:-}
EXTRA_OV=()
for _ov in $HYDRA_EXTRA; do EXTRA_OV+=("$_ov"); done
PROBE_EVERY=${PROBE_EVERY:-}
if [ -n "$PROBE_EVERY" ]; then
    case "$DATASET" in
        trifeatures)    EXTRA_OV+=(+probe_every_n_epochs="$PROBE_EVERY") ;;
        visionandtouch) EXTRA_OV+=(linear_probing_reg.every_n_epochs="$PROBE_EVERY") ;;
        sen1floods11)   echo "[job] sen1floods11 has no linear_probing block; PROBE_EVERY ignored" ;;
        *)              EXTRA_OV+=(linear_probing.every_n_epochs="$PROBE_EVERY") ;;
    esac
fi

# The run directory is `results/<timestamp>_<name>`, with `name` composed exactly as
# `utils.build_run_identity` does, so RESUME matches one cell and one seed only.
run_name() {
    local seed=$1 n aug gs
    if [ "$FAMILY" = "womm" ]; then
        n="${DATASET}_WoMM_${A}_${B}_${C}_fixedlbd"
    else
        aug="default"; [ "$B" != "-" ] && aug="$B"
        n="${DATASET}_MMSD_${A}_aug-${aug}"
    fi
    [ "$ARM" != "-" ] && n="${n}_${ARM}"
    gs=$(printf '%s\n' $HYDRA_EXTRA | sed -n 's/^+\?group_suffix=//p' | head -1)
    [ -n "$gs" ] && n="${n}_${gs}"
    printf '%s_s%s' "$n" "$seed"
}
find_ckpt() {
    ls -1t results/*_"$(run_name "$1")"/*/*/checkpoints/*.ckpt 2>/dev/null | head -1
}

# sbatch runs a copy of this script from its spool directory, so its own location
# only points at the repository when launched with bash.
if [ -z "${REPO_DIR:-}" ]; then
    REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    [ -f "$REPO_DIR/main_trifeatures.py" ] || REPO_DIR=${SLURM_SUBMIT_DIR:-$PWD}
fi
if [ ! -f "$REPO_DIR/main_trifeatures.py" ]; then
    echo "[job] '$REPO_DIR' is not the repository root: submit from it or set REPO_DIR"
    exit 1
fi

CONDA_ENV=${CONDA_ENV-multimodal}
if [ -n "$CONDA_ENV" ]; then
    CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null)}
    if [ -z "$CONDA_BASE" ]; then
        echo "[job] conda not found: set CONDA_BASE, or CONDA_ENV='' to use the current python"
        exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || exit 1
fi
if [ -n "${HF_HOME:-}" ]; then
    export SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence_transformers}
fi
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800

cd "$REPO_DIR"
pwd; hostname; date; echo "[job] python: $(command -v python)"

# One file per run holding its exit code, so the job fails when any seed fails
# instead of reporting success once the background runs have merely started.
STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cell_status.XXXXXX")
trap 'rm -rf "$STATUS_DIR"' EXIT

get_free_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'
}
throttle() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PAR" ]; do
        wait -n 2>/dev/null || sleep 30
    done
}

build_model_ov() {
    if [ "$FAMILY" = "womm" ]; then
        MODEL_OV=("${MODEL_PLUS}model=womm"
                  model.model.loss_kwargs.reconstruction="$A"
                  model.model.loss_kwargs.regularization="$B"
                  model.model.loss_kwargs.reg_weight="$C"
                  model.model.loss_kwargs.use_geco=False)
    else
        MODEL_OV=("${MODEL_PLUS}model=mmsd"
                  model.model.loss_kwargs.objective="$A"
                  model.model.loss_kwargs.out_dim=$PROTOS)
        # CREMA-D's 16 video and 25 audio tokens are perfect squares, which the mask
        # sampler would otherwise read as a 2-D grid.
        [ "$DATASET" = "crema_d_features" ] && MODEL_OV+=(+model.model.mask_kwargs.mode=span)
        if [ "$B" != "-" ] && [ "$DATASET" = "trifeatures" ]; then
            MODEL_OV+=("+data.data_module.augment=[$B,$B]")
        fi
    fi
}

launch() {
    local seed=$1
    build_model_ov
    local resume=()
    if [ "$RESUME" = "1" ]; then
        local ckpt wid
        ckpt=$(find_ckpt "$seed")
        if [ -z "$ckpt" ]; then
            echo "[launch] REFUSING $DATASET/$A/$B/$C arm=$ARM seed=$seed: no checkpoint for $(run_name "$seed")"
            echo "1" > "$STATUS_DIR/${SLUG}_s${seed}"
            return 1
        fi
        wid=$(basename "$(dirname "$(dirname "$ckpt")")")
        # Checkpoint names contain '=', which Hydra only accepts inside a quoted value.
        resume=("+ckpt_path='$ckpt'" "+wandb_id=$wid")
        echo "[launch] resume seed=$seed from $ckpt (wandb $wid)"
    fi
    export MASTER_PORT=$(get_free_port)
    echo "[launch] $DATASET/$FAMILY $A/$B/$C arm=$ARM seed=$seed port=$MASTER_PORT"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "    python $MAIN seed=$seed ${resume[*]} ${DATA_OV[*]} data.data_module.num_workers=$WORKERS ${MODEL_OV[*]} ${EXTRA_OV[*]} mode=train trainer.max_epochs=$EPOCHS"
        echo "0" > "$STATUS_DIR/${SLUG}_s${seed}"
        return 0
    fi
    (
    python "$MAIN" \
        seed=$seed \
        "${resume[@]}" \
        "${DATA_OV[@]}" \
        data.data_module.num_workers=$WORKERS \
        "${MODEL_OV[@]}" \
        "${EXTRA_OV[@]}" \
        mode="train" \
        trainer.max_epochs=$EPOCHS
    echo "$?" > "$STATUS_DIR/${SLUG}_s${seed}" ) &
    sleep 4
}

n_seeds=0; for seed in $SEEDS; do n_seeds=$((n_seeds + 1)); done
n_cells=0; for cell in $CELLS; do n_cells=$((n_cells + 1)); done
n=$((n_cells * n_seeds))
echo "[job] $DATASET arm=$ARM :: $n_cells cell(s) x $n_seeds seed(s) = $n runs, max $MAX_PAR concurrent"

for cell in $CELLS; do
    IFS=':' read -r FAMILY A B C <<< "$cell"
    B=${B:-'-'}; C=${C:-'-'}
    case "$FAMILY" in womm|mmsd) ;; *) echo "[job] unknown family '$FAMILY' in cell '$cell'"; exit 1 ;; esac
    SLUG=$(printf '%s' "$FAMILY-$A-$B-$C" | tr -c 'A-Za-z0-9.-' '_')
    echo "[job] --- cell $cell ---"
    for seed in $SEEDS; do
        throttle
        launch "$seed"
    done
    wait
    date
done

n_done=$(ls -1 "$STATUS_DIR" 2>/dev/null | wc -l)
n_bad=$(grep -Lx 0 "$STATUS_DIR"/* 2>/dev/null | wc -l)
echo "[job] $n runs launched, $n_done reported, $n_bad failed"
if [ "$n_bad" -gt 0 ] || [ "$n_done" -ne "$n" ]; then
    echo "[job] FAILING the job so it is not recorded as COMPLETED"
    for f in "$STATUS_DIR"/*; do
        [ -e "$f" ] && echo "[job]   $(basename "$f") exit=$(cat "$f")"
    done
    exit 1
fi
