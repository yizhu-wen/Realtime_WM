#!/bin/bash
# Train the real-time watermarking model.
#
# Expects WANDB_API_KEY in the environment when wandb.enabled is true in
# config/train.yaml. Add it to your shell profile once:
#
#   echo 'export WANDB_API_KEY=<your key>' >> ~/.bashrc && source ~/.bashrc
#
# Never commit the key to config/train.yaml.
set -euo pipefail

cd "$(dirname "$0")/.."

python train.py \
    -p config/process.yaml \
    -m config/model.yaml \
    -t config/train.yaml
