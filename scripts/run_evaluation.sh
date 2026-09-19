#!/usr/bin/env bash
# Evaluate one or more methods over the four corpora, then build the tables.
#
#   bash scripts/run_evaluation.sh                 # RT-SW only, both suites
#   METHODS="RTSW AudioSeal" bash scripts/run_evaluation.sh
#   SUITES="dl" JOBS=4 bash scripts/run_evaluation.sh
#
# Runs are resumable: each writes a checkpoint every 50 clips and picks up where
# it left off, so interrupting costs at most 50 clips.
set -euo pipefail
cd "$(dirname "$0")/../watermarking_model"

PY="${PY:-python}"
CKPT="${CKPT:-checkpoints/rtsw_ep20.pth.tar}"
METHODS="${METHODS:-RTSW}"
CORPORA="${CORPORA:-LibriSpeech-dev LJSpeech clone_xspeaker resynth_hifigan}"
SUITES="${SUITES:-dl paper}"
JOBS="${JOBS:-3}"

run_one () {   # method corpus suite
  local m=$1 c=$2 s=$3
  local dir="results/detect" ; [ "$s" = paper ] && dir="results/detect_paper"
  mkdir -p "$dir" results/log
  local out="$dir/${m}_${c}.npz"
  local extra=""; [ "$m" = RTSW ] && extra="--ckpt $CKPT"
  echo "[$s] $m / $c"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY detect_eval.py --method "$m" $extra --dataset "$c" --suite "$s" \
        --out "$out" >> "results/log/${s}_${m}_${c}.log" 2>&1
}
export -f run_one; export PY CKPT

for s in $SUITES; do
  for m in $METHODS; do
    for c in $CORPORA; do echo "$m $c $s"; done
  done
done | xargs -P "$JOBS" -n 3 bash -c 'run_one "$@"' _

# imperceptibility (SNR / PESQ) -- cheap, one encoder pass per clip
for m in $METHODS; do
  for c in $CORPORA; do
    mkdir -p results/imp
    extra=""; [ "$m" = RTSW ] && extra="--ckpt $CKPT"
    $PY imperceptibility.py --method "$m" $extra --dataset "$c" \
        --out "results/imp/${m}_${c}.npz" >> "results/log/imp_${m}_${c}.log" 2>&1 || true
  done
done

echo "building tables"
$PY detect_report.py --in results/detect       --imp results/imp --out results/evals
$PY detect_report.py --in results/detect_paper --imp results/imp --out results/evals/paper_suite
$PY merge_tables.py
echo "tables in watermarking_model/results/evals/"
