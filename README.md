# kinder-openpi

A fork of [OpenPI](https://github.com/Physical-Intelligence/openpi) (from Physical Intelligence) adapted for the [KinDER benchmark](https://github.com/irom-lab/KinDER).

---

## Overview

`kinder-openpi` extends upstream OpenPI with the following capabilities:

### Multi-Host TPU Training
- **Sophisticated device mesh construction** supporting intra-host FSDP, cross-host FSDP (whole-host grouping), and pure FSDP modes (`training/mh_sharding.py`)
- **Multi-host aware data loading**: per-host batch division, correct handling of cross-host FSDP edge cases, and checkpoint resumption via `dataset.skip(n)`

### Improved RLDS Data Loading Pipeline
- Full support for RLDS dataset formats
- **Easily extendable multi-dataset mixing** with automatic thread allocation (`dataloader/dataset_mixer.py`)
- Checkpoint-aware dataloader with batch counter saving/loading for seamless training resumption

---

## Installation

Clone the repository **with all submodules**:

```bash
git clone --recurse-submodules git@github.com:lihzha/kinder-openpi.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## Environment Setup

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies.

1. **Install `uv`** following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

2. **Sync the environment**:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## Training

### Step 1: Prepare Data (Convert to TFDS)

After obtaining the HDF5 trajectory file from replaying, set the path to the raw file:

```bash
export HDF5_FILE_PATH=<path_to_hdf5>
```

Build the TFDS dataset. First, follow the README in `third_party/rlds_dataset_builder` to set up the environment. Then run:

```bash
cd third_party/rlds_dataset_builder/<dataset_type>
conda activate rlds
tfds build --overwrite
```

where `<dataset_type>` is:
- `planning_threedim_dataset` for 3D tasks
- `planning_twodim_dataset` for 2D tasks

---

### Step 2: Launch Training

After building the TFDS dataset, place it under `data/planning_dataset`. Then launch training:

```bash
uv run scripts/train.py pi05_kinder_finetune --exp-name=kinder --fsdp-devices=8
```

Adjust `--exp-name` and `--fsdp-devices` as needed. Additional hyperparameters can be set via command line or in `src/kinder_openpi/training/config.py`. Training metrics are logged to W&B and checkpoints are saved to `./checkpoints` by default.

---

## Evaluation

### Step 1: Launch the Policy Server (Terminal 1)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_kinder_finetune \
  --policy.dir=checkpoints/<exp_name>/<epoch>
```

### Step 2: Run the Evaluation Script (Terminal 2)

Install the required packages:

```bash
pip install openpi_client tyro
```

Then run:

```bash
# 3D environments
python scripts/eval.py --use_overview_image --open-loop-horizon=8

# 2D environments
python scripts/eval.py --no-use_overview_image
```

### Step 3: Launch the Environment (Terminal 3)

Start the corresponding robot or simulation environment. See the [KinDER repo](https://github.com/irom-lab/KinDER) for details.
