#!/bin/bash
# Run all six downstream tasks across the three evaluation modes. Run from the
# repository root:
#   CKPT=checkpoints/latest.pt bash downstream/run_downstream.sh
set -e

CKPT=${CKPT:-checkpoints/latest.pt}
OUTDIR=${OUTDIR:-checkpoints/downstream}
TASKS="geology age composition crossmodal mare craters"

for task in $TASKS; do
    for mode in scratch linear finetune; do
        echo ">>> $task / $mode"
        python "downstream/task_${task}.py" \
            --checkpoint "$CKPT" --mode "$mode" --output-dir "$OUTDIR"
    done
done

# Few-shot evaluation (geology and age, 5- and 10-shot).
python downstream/run_fewshot_fast.py --output-dir "$OUTDIR"
