#!/bin/bash
# MG-MAE pretraining launcher. Run from the repository root.
#   bash pretraining/run_pretrain.sh
# Adjust NPROC / CUDA_VISIBLE_DEVICES to your machine.
set -e

NPROC=${NPROC:-2}
CKPT_DIR=${CKPT_DIR:-checkpoints}

torchrun --nproc_per_node="$NPROC" pretraining/train_mae.py \
    --ddp \
    --data-source mmap \
    --batch-size 64 --grad-accum 2 \
    --epochs 100 --lr 1.5e-4 --warmup-epochs 10 \
    --mask-ratio 0.75 \
    --crossmodal-prob 0.5 --contrast-weight 0.1 \
    --val-every 5 --val-samples 2000 --save-every 5 \
    --checkpoint-dir "$CKPT_DIR" --num-workers 7
