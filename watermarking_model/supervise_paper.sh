#!/bin/bash
# Keep the paper-suite evaluation running until every corpus is complete.
# WavMark is capped at 4 concurrent workers: measured at 26 s/clip alone, it
# degrades superlinearly under contention (5 workers 65 s/clip -> 16 workers
# 330 s/clip on the previous suite), so extra workers cost throughput.
cd /home/yizwen/research/Realtime_WM/watermarking_model
PY=/home/yizwen/.conda/envs/timbrewm/bin/python
TARGET=8; WM_MAX=4
CORPORA="LibriSpeech-dev LJSpeech clone_xspeaker resynth_hifigan"
complete () { local f="results/detect_paper/$1_$2.npz"; [ -f "$f" ] || return 1
  $PY -c "import numpy as np,sys; sys.exit(0 if bool(np.load('$f')['complete']) else 1)" 2>/dev/null; }
# RTSW's command line puts --ckpt between --method and --dataset, so the two
# flags must be matched separately; a single literal never matched for RTSW and
# the supervisor relaunched it on every pass.
running () { ps -eo comm,args | awk -v m="--method $1 " -v d="--dataset $2 " \
    '$1 ~ /^python/ && /suite paper/ && index($0,m) && index($0,d)' | grep -q .; }
nw () { ps -eo comm,args | awk '$1 ~ /^python/ && /suite paper/' | wc -l; }
nwm () { ps -eo comm,args | awk '$1 ~ /^python/ && /--method WavMark/ && /suite paper/' | wc -l; }
while true; do
  pending=0
  for M in WavMark SilentCipher Timbre AudioSeal RTSW; do for DS in $CORPORA; do
    complete "$M" "$DS" && continue
    pending=$((pending+1))
    running "$M" "$DS" && continue
    [ "$(nw)" -ge "$TARGET" ] && continue
    [ "$M" = WavMark ] && [ "$(nwm)" -ge "$WM_MAX" ] && continue
    rmdir "results/detect_paper/.lock_${M}_${DS}" 2>/dev/null
    echo "$(date -Is) launching $M/$DS (workers=$(nw))"
    setsid nohup /tmp/one_paper.sh "$M:$DS" >> results/log/paper_launch.log 2>&1 < /dev/null &
    sleep 20
  done; done
  [ "$pending" -eq 0 ] && { echo "$(date -Is) PAPER SUITE COMPLETE"; break; }
  sleep 60
done
