#!/bin/bash
# Regenerate the Excel workbook and LaTeX tables from whatever scores exist.
# Safe to run at any time: every detect_eval/imperceptibility run checkpoints,
# and the clip order is shuffled, so a partial run is a valid smaller-N sample.
cd /home/yizwen/research/Realtime_WM/watermarking_model
PY=/home/yizwen/.conda/envs/timbrewm/bin/python
OUT=results/evals
$PY detect_report.py --in results/detect --imp results/imp --out "$OUT" \
    --n_boot "${N_BOOT:-2000}" > "$OUT/refresh.log" 2>&1
rc=$?
# plain-text copy of both tables, for pasting into the paper
{
  echo "Generated $(date -Is)"
  echo "Sample sizes and completion state:"
  $PY - <<'PYEOF'
import numpy as np, glob, os, re
DS = ['LibriSpeech-dev','LJSpeech','clone_xspeaker','resynth_hifigan']
for kind in ("detect", "imp"):
    print(f"  [{kind}]")
    for f in sorted(glob.glob(f"results/{kind}/*.npz")):
        b = os.path.basename(f)[:-4]
        if not re.match(r'^(.*)_(%s)$' % '|'.join(map(re.escape, DS)), b): continue
        d = np.load(f)
        print(f"    {b:<34} n={int(d['used']):>5}  "
              f"{'complete' if bool(d['complete']) else 'PARTIAL'}")
PYEOF
  for f in distortion_comparison_exact1pct.tex distortion_comparison.tex imperceptibility.tex; do
    echo; echo "================================================================"
    echo "  $f"; echo "================================================================"; echo
    cat "$OUT/$f" 2>/dev/null
  done
} > "$OUT/results_tables.txt"
echo "$(date -Is) rc=$rc -> $OUT/results_tables.txt"
exit $rc
