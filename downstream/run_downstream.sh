#!/bin/bash
# Run all six downstream tasks across the three evaluation modes. Run from the
# repository root:
#   CKPT=checkpoints/latest.pt bash downstream/run_downstream.sh
#
# NUM_WORKERS controls DataLoader parallelism. Set NUM_WORKERS=0 inside ROCm /
# Singularity containers, where forked workers deadlock against the HIP context.
set -e

CKPT=${CKPT:-checkpoints/latest.pt}
OUTDIR=${OUTDIR:-checkpoints/downstream}
NUM_WORKERS=${NUM_WORKERS:-4}
TASKS="geology age composition crossmodal mare craters"

for task in $TASKS; do
    for mode in scratch linear finetune; do
        echo ">>> $task / $mode"
        python "downstream/task_${task}.py" \
            --checkpoint "$CKPT" --mode "$mode" --output-dir "$OUTDIR" \
            --num-workers "$NUM_WORKERS"
    done
done

# Few-shot evaluation (geology and age, 5- and 10-shot).
python downstream/run_fewshot_fast.py --output-dir "$OUTDIR" --num-workers "$NUM_WORKERS"
