#!/bin/bash
# Keep the machine full until every (method, corpus) detection run is complete.
#
# Replaces the xargs schedulers, which walked a fixed job list exactly once: a
# job that happened to be locked when the scheduler reached it was skipped and
# never retried, so six jobs sat idle while the GPU ran at half capacity. This
# re-scans instead, so a job is only finished when its checkpoint says so.
#
# Longest-remaining method first: SilentCipher and WavMark cost ~32 s/clip
# against Timbre and AudioSeal at ~12, so starting them late would leave them
# running alone for hours at the end.
cd /home/yizwen/research/Realtime_WM/watermarking_model
PY=/home/yizwen/.conda/envs/timbrewm/bin/python
TARGET=${TARGET:-12}
ORDER="SilentCipher WavMark AudioSeal Timbre"
CORPORA="LibriSpeech-dev LJSpeech clone_xspeaker resynth_hifigan"

complete () {  # $1=method $2=corpus
  local f="results/detect/$1_$2.npz"
  [ -f "$f" ] || return 1
  $PY -c "import numpy as np,sys; sys.exit(0 if bool(np.load('$f')['complete']) else 1)" 2>/dev/null
}
running () { ps -eo comm,args | awk -v p="--method $1 --dataset $2" '$1 ~ /^python/ && $0 ~ p' | grep -q .; }
nworkers () { ps -eo comm,args | awk '$1 ~ /^python/ && /detect_eval/' | wc -l; }

while true; do
  pending=0
  for M in $ORDER; do for DS in $CORPORA; do
    complete "$M" "$DS" && continue
    pending=$((pending+1))
    running "$M" "$DS" && continue
    [ "$(nworkers)" -ge "$TARGET" ] && continue
    # a lock with no live owner is stale (crash, kill, or a deferral) -- clear it
    rmdir "results/detect/.lock_${M}_${DS}" 2>/dev/null
    echo "$(date -Is) launching $M/$DS (workers=$(nworkers))"
    setsid nohup /tmp/one_detect3.sh "$M:$DS" >> results/log/supervisor_launch.log 2>&1 < /dev/null &
    sleep 20
  done; done
  [ "$pending" -eq 0 ] && { echo "$(date -Is) ALL DETECTION RUNS COMPLETE"; break; }
  sleep 60
done
