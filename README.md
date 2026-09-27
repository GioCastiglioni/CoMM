# ReMM: What to Regularize in Multimodal Self-Supervised Learning?

ReMM (**Re**gularized **M**ulti**M**odal learning) writes multimodal self-supervised objectives as

$$\mathcal{L}_{\text{ReMM}} = \mathcal{L}_{\text{align}} + \lambda\,\mathcal{L}_{\text{reg}},$$

where $\mathcal{L}_{\text{align}}$ aligns unimodal and multimodal embeddings across augmented views, and $\mathcal{L}_{\text{reg}}$ prevents collapse. [CoMM](https://arxiv.org/abs/2409.07402) is the special case of cosine alignment with an InfoNCE regularizer and $\lambda = 1$, and this repository reduces to it exactly. The study replaces the regularizer by other contrastive candidate sets and by Gaussian regularizers (SIGReg, VISReg), and measures the redundancy, uniqueness and synergy each one retains.

This repository is built on the code of CoMM, and keeps its name for now.

- [Installation](#installation)
- [Data](#data)
- [Running experiments](#running-experiments)
- [Experiments of the paper](#experiments-of-the-paper)
- [Outputs](#outputs)

## Installation

```sh
git clone <this repository> && cd CoMM
conda env create -f environment.yml
conda activate multimodal
```

Runs log to [Weights & Biases](https://wandb.ai) under the project `WoMM`. Run `wandb login` once, or set `WANDB_MODE=offline` (or `disabled`) to keep the logs local.

Every run writes its checkpoints under `results/` inside the repository, so place the clone on a filesystem with room for them.

## Data

All entry points read their dataset paths from `dataset/catalog.json`. Write it for your machine with:

```sh
python scripts/configure_catalog.py --root /path/to/data
```

which expects the layout below under `--root`. Any dataset stored elsewhere can be pointed at directly, e.g. `--path so2sat=/scratch/so2sat/arrays`. Use `--dry-run` to print the catalog and flag missing paths without writing it.

```
<root>/
├── trifeatures/                 generated on first use
├── multibench/
│   ├── mosi_data.pkl
│   ├── humor.pkl                UR-FUNNY
│   └── sarcasm.pkl              MUStARD
├── so2sat/
│   ├── geobench_so2sat.tortilla downloaded
│   └── arrays/                  written by scripts/convert_so2sat.py
├── hateful_memes/
│   ├── img/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── mmimdb/
    ├── split.json
    └── dataset/                 <id>.json and <id>.jpeg
```

**Bimodal Trifeatures** is rendered by `dataset/trifeatures.py` the first time it is instantiated. Generate it once before launching concurrent seeds, which would otherwise race to write the same files:

```sh
python -c "import json; from dataset.trifeatures import Trifeatures; \
           Trifeatures(json.load(open('dataset/catalog.json'))['trifeatures']['path'])"
```

**MultiBench** (MOSI, UR-FUNNY, MUStARD) uses the pre-extracted feature pickles:

```sh
bash scripts/fetch_multibench.sh /path/to/data/multibench mosi humor sarcasm
```

**So2Sat** (GeoBench) is distributed as a single `.tortilla` file at [hf.co/datasets/aialliance/so2sat](https://huggingface.co/datasets/aialliance/so2sat). Place it in `<root>/so2sat/` and convert it to memory-mappable arrays. The converter needs `numpy`, `pyarrow` and `rasterio`, which the training environment does not include, so run it from a separate environment. Pass `--name` if the downloaded file is not called `geobench_so2sat.tortilla`:

```sh
python scripts/convert_so2sat.py --root /path/to/data/so2sat
```

**Hateful Memes** is the official release of [Kiela et al. (2020)](https://arxiv.org/abs/2005.04790), extracted into `<root>/hateful_memes/`. The `test` split has no labels, so the linear probe is fitted on `train` and evaluated on `dev`.

**MM-IMDb** ([Arevalo et al., 2017](https://arxiv.org/abs/1702.01992)):

```sh
wget https://archive.org/download/mmimdb/mmimdb.tar.gz
tar xzf mmimdb.tar.gz -C /path/to/data
```

**Frozen pre-trained encoders.** Hateful Memes and MM-IMDb use a frozen CLIP ViT-B/32 and a multilingual Sentence-BERT, downloaded into the Hugging Face cache. Set `HF_HOME` to a location with enough space, and fill the cache once before launching concurrent seeds:

```sh
export HF_HOME=/path/to/hf_cache
python -c "import timm; timm.create_model('vit_base_patch32_clip_224.openai', pretrained=True); \
           from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('clip-ViT-B-32-multilingual-v1')"
```

## Running experiments

Every experiment goes through one launcher, [`scripts/run_cell.sh`](scripts/run_cell.sh). A **cell** is one configuration of the objective, and the launcher runs all the seeds of a cell concurrently on one GPU.

```sh
bash scripts/run_cell.sh <dataset> <family> <alignment> <regularizer> <lambda> "<seeds>" [arm]
```

For example, CoMM on the biased arm of Trifeatures with five seeds:

```sh
bash scripts/run_cell.sh trifeatures womm cosine neg-samples 1.0 "1 7 42 1234 2026" biased
```

The same line works with `sbatch` in place of `bash`. Submit from the repository root, and create the log directory first:

```sh
mkdir -p slurm_out
sbatch -J tri_comm_biased scripts/run_cell.sh trifeatures womm cosine neg-samples 1.0 "1 7 42 1234 2026" biased
```

The script requests 1 GPU, 32 CPUs, 200 GB of memory and 24 hours, which fits five concurrent seeds in every experiment below, except where noted there. Flags given to `sbatch` override these defaults, and cluster-specific options (partition, QoS, account) go there as well, e.g. `sbatch --partition=gpu --qos=normal scripts/run_cell.sh ...`.

**Datasets:** `trifeatures`, `mosi`, `humor` (UR-FUNNY), `sarcasm` (MUStARD), `so2sat`, `hateful_memes`, `mmimdb`. Trifeatures also takes the arm, `biased` (synergy) or `unbiased` (redundancy and uniqueness).

**Cells** are given as `<alignment> <regularizer> <lambda>`:

| Alignment | Regularizer | λ | |
|---|---|---|---|
| `cosine` | `neg-samples` | `1.0` | CoMM (NT-Xent candidate set) |
| `cosine` | `neg-samples-dcl` | `1.0` | DCL candidate set |
| `cosine` | `neg-samples-cross` | `1.0` | CLIP candidate set |
| `cosine` | `neg-samples-dcl-cross` | `1.0` | CLIP candidate set without the positive |
| `mse` | `sigreg` | `0.05` | SIGReg |
| `mse` | `sigreg-permask` | `0.05` | SIGReg applied per mask |
| `mse` | `visreg` | `9.0` | VISReg |
| `mse` | `neg-samples` | `1.0` | alignment control |
| `mse` | `neg-samples-dcl` | `1.0` | alignment control (diverges) |
| `mse` | `mix-sigreg-neg-samples` | `1.0` | SIGReg and Neg-Samples on the whole embedding |
| `mse` | `split-sigreg-neg-samples` | `1.0` | Neg-Samples on one half of the embedding, SIGReg on the other |

The λ values reproduce the regularization-to-alignment ratio of each original method. The composite regularizers weight their two terms by `mix_rho` (0.95 in `configs/model/womm.yaml`).

**Several cells in one job.** Set `CELLS` to a list of `womm:<alignment>:<regularizer>:<lambda>` entries, or to the groups below, and pass `-` for the positional cell. Cells run one after another.

| Group | Cells |
|---|---|
| `@negsamples` | the four `cosine` Neg-Samples variants |
| `@negsamples-a` | `neg-samples`, `neg-samples-dcl` |
| `@negsamples-b` | `neg-samples-cross`, `neg-samples-dcl-cross` |
| `@gauss` | `sigreg`, `sigreg-permask`, `visreg` |
| `@align` | the two `mse` alignment controls |
| `@composite` | `mix-sigreg-neg-samples`, `split-sigreg-neg-samples` |

```sh
CELLS="@negsamples @gauss" sbatch -J mosi scripts/run_cell.sh mosi - - - - "1 7 42 1234 2026"
```

**Environment variables:**

| Variable | Default | |
|---|---|---|
| `CONDA_ENV` | `multimodal` | conda environment to activate; `''` uses the python already on `PATH` |
| `CONDA_BASE` | `conda info --base` | conda installation, if `conda` is not on `PATH` inside the job |
| `REPO_DIR` | script's parent, or the `sbatch` submit directory | repository root |
| `HF_HOME` | Hugging Face default | cache of the frozen encoders |
| `PROBE_EVERY` | every epoch | period of the intermediate linear probes, in epochs |
| `HYDRA_EXTRA` | — | extra Hydra overrides, space separated |
| `EPOCHS` | `100` | pre-training epochs |
| `MAX_PAR` | `5` | seeds run at once |
| `WORKERS` | 6 for image-text datasets, 4 otherwise | data loader workers per seed |
| `RESUME` | `0` | `1` continues every run from its last checkpoint |
| `DRY_RUN` | `0` | `1` prints the commands without running them |

The reported numbers always come from the linear probe of the final weights, run by `trainer.test`. The intermediate probes only draw learning curves, and are expensive on Trifeatures, where each epoch probes 4 tasks × 3 masks. Setting `PROBE_EVERY` above the number of epochs skips them without changing any result.

Learning rate, weight decay and batch size come from each dataset's train config (`configs/train_*.yaml`). Change them through `HYDRA_EXTRA` only for an ablation, together with a `+group_suffix=<name>`, which is appended to the run name so that those runs stay apart from the reference ones.

**Resuming.** If a job reaches its time limit, resubmit it with `RESUME=1` and the same arguments. Each seed continues from its last checkpoint and logs into its original W&B run. The launcher refuses a seed whose checkpoint it cannot find, rather than retraining it from scratch.

## Experiments of the paper

All experiments use the seeds `1 7 42 1234 2026`, and are launched from the repository root:

```sh
SEEDS="1 7 42 1234 2026"
mkdir -p slurm_out
```

**Bimodal Trifeatures.** Main table, alignment control and composite regularizers, on both arms:

```sh
for arm in biased unbiased; do
  for group in @negsamples-a @negsamples-b @gauss @align @composite; do
    CELLS="$group" PROBE_EVERY=1000 sbatch -J tri_${group#@}_${arm} \
      scripts/run_cell.sh trifeatures - - - - "$SEEDS" $arm
  done
done
```

Uniqueness and redundancy are read from the `unbiased` arm and synergy from the `biased` one.

**MultiBench and So2Sat.** MOSI and UR-FUNNY are reported at the batch size of 64 of the batch-size ablation below, dropping the last incomplete batch of each epoch. MUStARD and So2Sat use their default protocol.

```sh
for ds in mosi humor; do
  CELLS="@negsamples @gauss" PROBE_EVERY=1000 \
  HYDRA_EXTRA="data.data_module.batch_size=64 +data.data_module.drop_last=true +group_suffix=bs64" \
    sbatch -J ${ds}_bs64 scripts/run_cell.sh $ds - - - - "$SEEDS"
done
CELLS="@negsamples @gauss" PROBE_EVERY=1000 sbatch -J sarcasm scripts/run_cell.sh sarcasm - - - - "$SEEDS"
for group in @negsamples-a @negsamples-b @gauss; do
  CELLS="$group" PROBE_EVERY=1000 sbatch -J so2sat_${group#@} scripts/run_cell.sh so2sat - - - - "$SEEDS"
done
```

**Batch-size ablation**, on MOSI and UR-FUNNY, at the learning rate of each dataset's config. `B=64` is the run of the main table above. The cost of a cell grows as the batch shrinks, since an epoch takes more steps, so the cells are grouped differently per batch size to fit 24-hour jobs. Measured on one H100 with five concurrent seeds:

| Batch size | MOSI | UR-FUNNY |
|---|---|---|
| 4 | 7 cells in one job (~16 h) | one cell per job (12–15 h each) |
| 16 | 7 cells in one job (~4.5 h) | `@negsamples` (~13 h) and `@gauss` (~11 h) in two jobs |
| 64 | 7 cells in one job (~2 h) | 7 cells in one job (~7 h) |
| 256 | 7 cells in one job (~1.5 h) | 7 cells in one job (~2.5 h) |
| 1024 | — | 7 cells, seeds split 3 + 2 over two jobs |

```sh
# bs <dataset> <batch size> <cells> <job name> [seeds] [extra overrides]
bs() {
  CELLS="$3" PROBE_EVERY=1000 \
  HYDRA_EXTRA="data.data_module.batch_size=$2 +data.data_module.drop_last=true ${6:-+group_suffix=bs$2}" \
    sbatch -J "$4" scripts/run_cell.sh "$1" - - - - "${5:-$SEEDS}"
}

for B in 4 16 256; do bs mosi $B "@negsamples @gauss" mosi_bs$B; done
bs humor 256 "@negsamples @gauss" humor_bs256
bs humor 16 "@negsamples" humor_bs16_negs
bs humor 16 "@gauss" humor_bs16_gauss
for cell in cosine:neg-samples:1.0 cosine:neg-samples-dcl:1.0 cosine:neg-samples-cross:1.0 \
            cosine:neg-samples-dcl-cross:1.0 mse:sigreg:0.05 mse:sigreg-permask:0.05 mse:visreg:9.0; do
  bs humor 4 "womm:$cell" "humor_bs4_$(echo $cell | cut -d: -f2)"
done
```

At `B=1024` the five seeds do not fit together in the memory of one 80 GB GPU, so they are split over two jobs. UR-FUNNY runs there at the reference learning rate, and again at the learning rate scaled by the square-root rule, $10^{-3}\sqrt{1024/64} = 4\times10^{-3}$:

```sh
for seeds in "42 1234 2026" "1 7"; do
  bs humor 1024 "@negsamples @gauss" humor_bs1024 "$seeds"
  bs humor 1024 "@negsamples @gauss" humor_sqrtlr_bs1024 "$seeds" "optim.lr=4e-3 +group_suffix=sqrtlr_bs1024"
done
```

**Hateful Memes and MM-IMDb** (CoMM's protocol: frozen encoders, 15% token masking on text):

```sh
for ds in hateful_memes mmimdb; do
  for group in @negsamples-a @negsamples-b @gauss; do
    CELLS="$group" PROBE_EVERY=1000 sbatch -J ${ds}_${group#@} scripts/run_cell.sh $ds - - - - "$SEEDS"
  done
done
```

MM-IMDb decodes its full-resolution posters on the CPU, which makes it far slower than the other datasets. If its jobs reach the time limit, run fewer seeds per job with more loader workers each (e.g. `MAX_PAR=1 WORKERS=28` with one seed), and continue the runs across jobs with `RESUME=1`.

## Outputs

Every run writes to `results/<timestamp>_<run name>/`, where the run name encodes the dataset, the cell, the arm, the `group_suffix` and the seed, e.g. `trifeatures_WoMM_mse_sigreg_0.05_fixedlbd_biased_s42`. The last checkpoint is kept for resuming. The probe metrics are logged to W&B as `<metric>_<task>_<mask>`, where the mask is `both` (joint embedding), `mod1` or `mod2`:

| Dataset | Reported metric |
|---|---|
| Trifeatures | `acc1_share_both` (R), `acc1_unique1_both`, `acc1_unique2_both` (U), `acc1_synergy_both` (S) |
| MOSI, UR-FUNNY, MUStARD, So2Sat | `acc1_<dataset>_both`, with `<dataset>` as passed to the launcher |
| Hateful Memes | `roc_auc_hateful_memes_both` |
| MM-IMDb | `f1_mean_mmimdb_both` (macro F1), `f1_weighted_mmimdb_both` (weighted F1) |

## Acknowledgements

This code builds on the implementation of CoMM, whose original notebooks remain in [`demo/`](demo/):

    @inproceedings{dufumier_castillo2025,
        title={What to align in multimodal contrastive learning?},
        author={Dufumier, Benoit and Castillo-Navarro, Javiera and Tuia, Devis and Thiran, Jean-Philippe},
        booktitle={International Conference on Learning Representations},
        year={2025}
    }
