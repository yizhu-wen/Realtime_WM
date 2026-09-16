#!/bin/bash
# Half the WavMark workers are SIGSTOPped to cut GPU contention: sixteen
# concurrent workers ran at ~330 s/clip for 172 clips/h aggregate, where five
# managed 65 s/clip for 277 clips/h -- the work does not parallelise, it
# queues. This resumes the paused half once the running half is done, so the
# pause is a scheduling decision rather than a job that never finishes.
cd /home/yizwen/research/Realtime_WM/watermarking_model
while true; do
  sleep 120
  RUNNING=$(ps -eo stat,args | awk '/venv-wm\/bin\/python detect_eval/ && $1 !~ /^T/' | wc -l)
  PAUSED=$(ps -eo stat,args | awk '/venv-wm\/bin\/python detect_eval/ && $1 ~ /^T/' | wc -l)
  if [ "$PAUSED" -gt 0 ] && [ "$RUNNING" -le 2 ]; then
    echo "$(date -Is) running=$RUNNING paused=$PAUSED -> resuming all paused workers"
    for p in $(ps -eo pid,stat,args | awk '/venv-wm\/bin\/python detect_eval/ && $2 ~ /^T/ {print $1}'); do
      kill -CONT "$p" 2>/dev/null
    done
  fi
  [ "$RUNNING" -eq 0 ] && [ "$PAUSED" -eq 0 ] && { echo "$(date -Is) all workers finished"; break; }
done
