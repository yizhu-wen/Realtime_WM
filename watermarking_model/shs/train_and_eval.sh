#!/bin/bash
# Full training, then the distortion evaluation on the final checkpoint.
# Launch detached:  setsid nohup ./shs/train_and_eval.sh > LOG 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/yizwen/.conda/envs/timbrewm/bin/python
TRAIN_CFG="${TRAIN_CFG:-config/train.yaml}"
N_ITEMS="${N_ITEMS:-200}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

echo "### TRAINING START $(date -Is)"
$PY train.py -p config/process.yaml -m config/model.yaml -t "$TRAIN_CFG"
rc=$?
echo "### TRAINING EXIT $rc $(date -Is)"
[ $rc -ne 0 ] && { echo "### training failed, skipping evaluation"; exit $rc; }

CKPT_DIR=$($PY -c "import yaml,os;print(os.path.join(yaml.safe_load(open('$TRAIN_CFG'))['path']['ckpt'],'pth'))")
CKPT=$(ls -t "$CKPT_DIR"/none-conv2_*.pth.tar 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "### no checkpoint found, skipping evaluation"; exit 1; fi
echo "### EVALUATION START $(date -Is) on $CKPT"
$PY evaluate.py --ckpt "$CKPT" -t "$TRAIN_CFG" --n_items "$N_ITEMS"
echo "### EVALUATION EXIT $? $(date -Is)"
