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
import numpy as np, glob, os, re, collections
# Report MERGED totals per (method, corpus). A corpus may be split across
# <Method>_<Corpus>__shardN.npz files; listing them separately made WavMark's
# 2700-clip LJSpeech read as 1375.
DS = ['LibriSpeech-dev', 'LJSpeech', 'clone_xspeaker', 'resynth_hifigan']
TOT = {'LibriSpeech-dev': 2303, 'LJSpeech': 2700,
       'clone_xspeaker': 1624, 'resynth_hifigan': 2000}
for kind in ("detect", "imp"):
    print(f"  [{kind}]")
    agg = collections.defaultdict(lambda: [0, 0, True])   # used, shards, complete
    for f in sorted(glob.glob(f"results/{kind}/*.npz")):
        b = re.sub(r"__shard\d+$", "", os.path.basename(f)[:-4])
        m = re.match(r"^(.*)_(%s)$" % "|".join(map(re.escape, DS)), b)
        if not m:
            continue
        d = np.load(f)
        e = agg[(m.group(1), m.group(2))]
        e[0] += int(d["used"]); e[1] += 1
        e[2] = e[2] and bool(d["complete"])
    for (meth, ds), (used, nsh, done) in sorted(agg.items()):
        tot = TOT[ds]
        shards = f", {nsh} shards" if nsh > 1 else ""
        state = "complete" if done and used >= tot * 0.97 else "PARTIAL"
        print(f"    {meth}_{ds:<20} n={used:>5}/{tot:<5} {state}{shards}")
PYEOF
  for f in distortion_comparison_exact1pct.tex distortion_comparison.tex imperceptibility.tex; do
    echo; echo "================================================================"
    echo "  $f"; echo "================================================================"; echo
    cat "$OUT/$f" 2>/dev/null
  done
} > "$OUT/results_tables.txt"
echo "$(date -Is) rc=$rc -> $OUT/results_tables.txt"
exit $rc
