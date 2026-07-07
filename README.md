<h1 align="center">Moonstone</h1>

<p align="center">
  <b>A Multimodal Foundation Model and Benchmark for Lunar Remote Sensing</b><br>
  Ayush Prasad · Swarnalee Mazumder · <b>ECCV 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.03644"><img src="https://img.shields.io/badge/arXiv-2607.03644-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://ayushprasad.com/projects/moonstone/"><img src="https://img.shields.io/badge/Project-Page-3f5a7d" alt="Project page"></a>
  <a href="https://huggingface.co/datasets/ayushprd/Moonstone"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Moonstone-ffcc00" alt="Hugging Face dataset"></a>
  <a href="https://github.com/ayushprd/Moonstone"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white" alt="Code"></a>
  <img src="https://img.shields.io/badge/license-MIT-informational" alt="License">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.03644"><b>📄 Paper</b></a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/pdf/2607.03644"><b>PDF</b></a> &nbsp;·&nbsp;
  <a href="https://ayushprasad.com/projects/moonstone/"><b>🌐 Project&nbsp;page</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/ayushprd/Moonstone/tree/main/pretraining"><b>🤗 Pretraining&nbsp;data</b></a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/ayushprd/Moonstone/tree/main/benchmark"><b>🤗 Benchmark</b></a>
</p>

---

This repository contains the code for **Moonstone**, accepted at **ECCV 2026**:
the data-preparation pipeline, the MG-MAE (Modality-Grouped Masked Autoencoder)
pretraining and inference code, and the six-task downstream benchmark.

## Links

| Resource | Location |
|----------|----------|
| 📄 Paper (arXiv) | https://arxiv.org/abs/2607.03644 |
| 🌐 Project page | https://ayushprasad.com/projects/moonstone/ |
| 💻 Code | https://github.com/ayushprd/Moonstone |
| 🤗 Dataset (root) | https://huggingface.co/datasets/ayushprd/Moonstone |
| &nbsp;&nbsp;↳ Pretraining data | https://huggingface.co/datasets/ayushprd/Moonstone/tree/main/pretraining |
| &nbsp;&nbsp;↳ Benchmark data | https://huggingface.co/datasets/ayushprd/Moonstone/tree/main/benchmark |

The dataset on Hugging Face is split into a `pretraining/` part (z-scored
memory-mapped arrays + source GeoTIFFs) and a `benchmark/` part
(`lunar_patches_v4.h5` + label rasters).

## Overview

Moonstone assembles 28 channels from seven instrument families across five lunar
missions onto a common 128 pixels-per-degree (~237 m/pixel) equirectangular grid,
organized into seven physical modality groups (surface, thermal, spectral M3,
gravity, radar, hapke, composition). MG-MAE pretrains a shared ViT encoder over
per-group convolutional tokenizers with attention masking for missing modalities,
coverage-adaptive masking, and spectral-continuity regularization.

## Install

```bash
pip install -e .
```

This installs the dependencies and puts the shared modules (`config`,
`lunar_dataset`, `lunar_mae_v2`, `downstream_base`, `downstream_dataset`) on the
path, so the pipeline, pretraining, inference, and downstream scripts can be run
from anywhere.

## Layout

```
config.py, lunar_dataset.py, lunar_mae_v2.py   shared model + data modules
downstream_base.py, downstream_dataset.py      shared downstream infrastructure
channel_stats.json                             per-channel normalization statistics

data_preparation/   15-step pipeline (steps 01-15) + fix_minirf + download_craters
pretraining/        train_mae.py (MG-MAE training) + run_pretrain.sh
inference/          eval_mae.py (reconstruction MSE + visualizations)
downstream/         six task_*.py + run_fewshot_fast.py + run_downstream.sh
```

## Data preparation

Reconstructs the dataset from public NASA PDS / USGS / ODE archives. The pipeline
runs in numeric order; each step writes into `data/` and `output/`.

```bash
python data_preparation/step01_download.py          # base layers (WAC, LOLA, SLDEM, Diviner)
python data_preparation/step02_align.py             # reproject to 128 ppd grid
python data_preparation/step03_derive.py            # slope, roughness
python data_preparation/step08_ode_query.py         # query PDS ODE for M3 products
python data_preparation/step09_m3_download.py       # download M3 (step09c filters nighttime)
python data_preparation/step10_m3_mosaic.py         # mosaic M3 strips
python data_preparation/step13_align_new_datasets.py  # GRAIL, Mini-RF, WAC Hapke, Clementine, LP GRS
python data_preparation/fix_minirf.py               # log1p transform for Mini-RF outliers
python data_preparation/step14_build_v4.py          # 28-channel HDF5
python data_preparation/step15_build_mmap.py --normalize   # z-scored mmap arrays for training
```

Alternatively, download the prepared arrays from the Hugging Face `pretraining/`
and `benchmark/` folders and skip the pipeline.

## Pretraining

```bash
bash pretraining/run_pretrain.sh
```

This launches `pretraining/train_mae.py` with the paper configuration (100 epochs,
effective batch 256, mask ratio 0.75, cross-modal masking 0.5, contrastive weight
0.1). Checkpoints are written to `checkpoints/`.

## Inference

Per-group reconstruction MSE and reconstruction figures from a checkpoint:

```bash
python inference/eval_mae.py --checkpoint checkpoints/latest.pt \
    --n-samples 500 --visualize --n-vis 4
```

## Downstream evaluation

Six tasks (geology, age, composition, cross-modal, mare, craters), each in
`scratch`, `linear`, or `finetune` mode:

```bash
python downstream/task_geology.py --checkpoint checkpoints/latest.pt --mode linear
```

Run the full benchmark plus few-shot:

```bash
CKPT=checkpoints/latest.pt bash downstream/run_downstream.sh
```

On AMD ROCm inside a Singularity/Apptainer container, pass `--num-workers 0`
(forked DataLoader workers can deadlock against the HIP context):

```bash
python downstream/task_geology.py --checkpoint checkpoints/latest.pt --mode linear --num-workers 0
# or for the whole suite:
NUM_WORKERS=0 CKPT=checkpoints/latest.pt bash downstream/run_downstream.sh
```

## Citation

If you use the Moonstone dataset, model, or benchmark, please cite:

```bibtex
@article{prasad2026moonstone,
  title   = {Moonstone: A Multimodal Foundation Model and Benchmark
             for Lunar Remote Sensing},
  author  = {Prasad, Ayush and Mazumder, Swarnalee},
  journal = {arXiv preprint arXiv:2607.03644},
  year    = {2026}
}
```

## License

Released under the MIT License. Source data is derived from public NASA PDS,
USGS, and ISRO archives.
