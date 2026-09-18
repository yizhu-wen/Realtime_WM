"""Append the paper-suite rows to the main distortion table as one table.

The two suites were each calibrated on their own pooled negatives, so simply
concatenating the .tex files would put two different operating points in one
table. Here every method gets ONE threshold, pooled over the negatives of all
reported conditions in both suites, and every row is recomputed against it.

AUC is exact, not bootstrapped: the scores are small integers, so
P(pos > neg) + 0.5 P(pos == neg) follows from the two histograms in O(bins).
That matches sklearn to 1e-12 and is ~100x faster, which matters because the
LaTeX tables carry point estimates only -- the bootstrap exists solely for the
workbook's CI columns.
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_report import (DL_ROWS, PAPER_ROWS, METHODS, DATASETS,  # noqa: E402
                           TEX_HEAD, TEX_TAIL, TARGET_FPR, fmt)

SUITES = [("results/detect", DL_ROWS), ("results/detect_paper", PAPER_ROWS)]

# Rows scored and stored, but not reported in the combined table. Removing a
# name here restores it; nothing needs re-running.
EXCLUDE = {"gaussian_noise_20", "speech_mix_-15dB", "duck", "boost",
           "resample_8k"}

# Rows forced to the end, in this order.
LAST = ["phone_call"]


def load_suite(indir, rows):
    """{method: {row: (pos, neg)}} with shards merged and corpora pooled."""
    import glob, collections
    parts = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        base = re.sub(r"__shard\d+$", "", os.path.basename(f)[:-4])
        if base.startswith("_"):
            continue
        m = re.match(r"^(.*)_(%s)$" % "|".join(map(re.escape, DATASETS)), base)
        if not m:
            continue
        d = np.load(f, allow_pickle=True)
        meth = str(d["method"])
        for _, k in rows:
            if f"pos_{k}" not in d.files:
                continue
            n = min(len(d[f"pos_{k}"]), len(d[f"neg_{k}"]))
            parts[meth][k].append((d[f"pos_{k}"][:n], d[f"neg_{k}"][:n]))
    out = {}
    for meth, byrow in parts.items():
        out[meth] = {k: (np.concatenate([p for p, _ in v]),
                         np.concatenate([n for _, n in v]))
                     for k, v in byrow.items()}
    return out


def auc_exact(pos, neg, nb):
    hp = np.bincount(pos.astype(int), minlength=nb + 1).astype(float)
    hn = np.bincount(neg.astype(int), minlength=nb + 1).astype(float)
    cn = np.concatenate(([0.0], np.cumsum(hn)))
    return (float((hp * cn[:-1]).sum()) + 0.5 * float((hp * hn).sum())) \
        / (hp.sum() * hn.sum())




AUX_TEX = r"""% put this in your preamble
\newcommand{\aux}[1]{{\scriptsize{\textcolor{gray}{#1}}}}

\begin{table*}[t]
    \centering
    \caption{
        Decoding evaluation results under different audio distortions.
        Acc. (\aux{TPR/FPR}) is the bit accuracy (and the true- and
        false-positive rates at a threshold calibrated to a 1\% false-positive
        rate on the pooled negatives). AUC is the area under the ROC curve.
    }
    \label{tab:wm_robustness}
    \vspace{4pt}
    \resizebox{1.0\linewidth}{!}{
        \begin{tabular}{l *{2}{l} *{2}{l} *{2}{l} *{2}{l} *{2}{l}}
        \toprule
        & \multicolumn{2}{l}{\textbf{RT-SW (Ours)}}
        & \multicolumn{2}{l}{\textbf{AudioSeal}}
        & \multicolumn{2}{l}{\textbf{WavMark}}
        & \multicolumn{2}{l}{\textbf{Timbre}}
        & \multicolumn{2}{l}{\textbf{SilentCipher}} \\
        \cmidrule(rr){2-3} \cmidrule(rr){4-5} \cmidrule(rr){6-7} \cmidrule(rr){8-9} \cmidrule(rr){10-11}
        \multicolumn{1}{c}{Distortion}
        & Acc. \aux{TPR/FPR} & AUC & Acc. \aux{TPR/FPR} & AUC
        & Acc. \aux{TPR/FPR} & AUC & Acc. \aux{TPR/FPR} & AUC
        & Acc. \aux{TPR/FPR} & AUC \\
        \cmidrule(rr){1-1} \cmidrule(rr){2-3} \cmidrule(rr){4-5} \cmidrule(rr){6-7} \cmidrule(rr){8-9} \cmidrule(rr){10-11}
"""

AUX_TAIL = r"""        \bottomrule
        \end{tabular}
    }
    \vspace{-0.2cm}
\end{table*}
"""


def write_aux_style(path, rows_all, data, thr, payload, nd=3):
    """AudioSeal-style layout: Acc with TPR/FPR as small grey \aux text, one
    AUC column per method, the strict per-row winner in bold, and an average
    row. Bolding is skipped on ties, which are common here because three
    baselines sit at 1.000 on most rows."""
    def cell(meth, k):
        p, n = data[meth][k]
        t, g, nb = thr[meth]
        rej = lambda x: float(np.mean(x >= t) + g * np.mean(x == t - 1))
        return p.mean() / nb, rej(p), rej(n), auc_exact(p, n, nb)

    body = []
    for label, k in rows_all:
        vals = {m: cell(m, k) for m in METHODS if k in data.get(m, {})}
        best = max(v[0] for v in vals.values())
        strict = sum(1 for v in vals.values() if v[0] >= best - 1e-9) == 1
        cells = []
        for m in METHODS:
            if m not in vals:
                cells.append("-- & --"); continue
            a, tp, fp, au = vals[m]
            b = r"\bf " if (strict and a >= best - 1e-9) else ""
            cells.append(f"{b}{a:.{nd}f} \\aux{{{tp:.{nd}f}/{fp:.{nd}f}}} & {b}{au:.{nd}f}")
        body.append(f"        {label} & " + " & ".join(cells) + r" \\")

    open(path, "w").write(AUX_TEX + "\n".join(body) + "\n" + AUX_TAIL)
    print(f"wrote {path}  ({len(rows_all)} rows)")


def main():
    payload = {"RT-SW": 10, "AudioSeal": 16, "WavMark": 16,
               "Timbre": 10, "SilentCipher": 40}
    data, rows_all = {}, []
    for indir, rows in SUITES:
        s = load_suite(indir, rows)
        present = [r for r in rows if r[1] not in EXCLUDE
                   and any(r[1] in s.get(m, {}) for m in METHODS)]
        rows_all += present
        for meth, byrow in s.items():
            data.setdefault(meth, {}).update(byrow)
        print(f"  {indir}: {len(present)} rows, {len(s)} methods")

    # phone call reads as the conclusion of the table, so it goes last
    rows_all = ([r for r in rows_all if r[1] not in LAST]
                + [r for k in LAST for r in rows_all if r[1] == k])

    lines = []
    thr = {}
    for meth in METHODS:
        nb = payload[meth]
        pooled = np.concatenate([data[meth][k] [1] for _, k in rows_all
                                 if k in data.get(meth, {})])
        t = next((i for i in range(nb + 1)
                  if float(np.mean(pooled >= i)) <= TARGET_FPR), nb + 1)
        above = float(np.mean(pooled >= t))
        at = float(np.mean(pooled == t - 1)) if t >= 1 else 0.0
        g = 0.0 if at <= 0 else min(1.0, max(0.0, (TARGET_FPR - above) / at))
        thr[meth] = (t, g, nb)
        print(f"  {meth:<13} payload {nb:>2}  threshold >= {t}  "
              f"pooled FPR {above:.4f} on {len(pooled)} negatives  "
              f"-> randomised gamma {g:.3f}")

    for label, k in rows_all:
        parts = [label]
        for meth in METHODS:
            if k not in data.get(meth, {}):
                parts.append("& -- & -- & --"); continue
            p, n = data[meth][k].__iter__(), None
            p, n = data[meth][k]
            t, g, nb = thr[meth]
            rej = lambda s: float(np.mean(s >= t) + g * np.mean(s == t - 1))
            parts.append(f"& {fmt(p.mean()/nb)} & {fmt(rej(p))}/{fmt(rej(n), 4)}"
                         f" & {fmt(auc_exact(p, n, nb))}")
        lines.append("\n".join(parts) + r" \\")

    out = "results/evals/distortion_comparison_exact1pct_combined.tex"
    open(out, "w").write(TEX_HEAD + "\n\n".join(lines) + TEX_TAIL)
    print(f"\nwrote {out}  ({len(rows_all)} rows)")
    write_aux_style("results/evals/distortion_comparison_aux_style.tex",
                    rows_all, data, thr, payload)
    return rows_all, data, thr, payload


if __name__ == "__main__":
    rows_all, data, thr, payload = main()
    w = f"{'distortion':<22}" + "".join(f"{m:>26}" for m in METHODS)
    print("\n" + w); print("-" * len(w))
    for label, k in rows_all:
        row = f"{label:<22}"
        for meth in METHODS:
            if k not in data.get(meth, {}):
                row += f"{'--':>26}"; continue
            p, n = data[meth][k]; t, g, nb = thr[meth]
            rej = lambda s: float(np.mean(s >= t) + g * np.mean(s == t - 1))
            row += f"  {p.mean()/nb:.3f} {rej(p):.3f}/{rej(n):.4f} {auc_exact(p,n,nb):.3f}".rjust(26)
        print(row)
