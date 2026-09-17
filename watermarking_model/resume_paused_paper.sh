#!/bin/bash
# Four paper-suite workers are SIGSTOPped to cut CPU oversubscription: the
# suite spawns ten ffmpeg subprocesses per clip (five codecs x positive and
# negative), so twelve workers drove load to 56 on 24 cores. Their state and
# checkpoints are intact; this resumes them as active slots free, so the pause
# is scheduling rather than an abandoned job.
cd /home/yizwen/research/Realtime_WM/watermarking_model
TARGET=8
while true; do
  sleep 120
  A=$(ps -eo stat,args | grep "[d]etect_eval.py" | grep "suite paper" | awk '$1 !~ /^T/' | wc -l)
  P=$(ps -eo pid,stat,args | grep "[d]etect_eval.py" | grep "suite paper" | awk '$2 ~ /^T/ {print $1}')
  [ -z "$P" ] && { [ "$A" -eq 0 ] && { echo "$(date -Is) all workers done"; break; }; continue; }
  if [ "$A" -lt "$TARGET" ]; then
    p=$(echo "$P" | head -1)
    kill -CONT "$p" && echo "$(date -Is) resumed pid $p (active was $A)"
  fi
done
