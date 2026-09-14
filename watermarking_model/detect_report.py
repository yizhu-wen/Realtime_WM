"""Aggregate the raw per-utterance scores from detect_eval.py.

Reads every ``<Method>_<Dataset>.npz`` under --in and produces

  * an Excel workbook: one sheet per evaluation corpus with bit accuracy
    (mean, std, 95% CI), AUC (bootstrap CI), and TPR/FPR at a calibrated
    threshold, plus a Summary sheet averaged over the four corpora;
  * the LaTeX body for tab:distortion_comparison.

Detection statistic
-------------------
The number of payload bits recovered. Positives are watermarked clips scored
against the embedded message; negatives are the same clips *unwatermarked*,
scored against an independent random message. Under H0 the statistic is
Binomial(n_bits, 0.5), so the negatives are an empirical null.

Threshold
---------
One threshold per method, not per cell. A deployed detector has a single
operating point, and a per-cell threshold would be fitted to ~2300 negatives
whose 1% quantile rests on ~23 samples. Pooling every negative a method
produced (4 corpora x 10 channels, ~86k draws) resolves that quantile far
better, and unwatermarked audio is close to Binomial(n, 0.5) whatever channel
it passed through. The threshold is the smallest integer whose pooled FPR is
at or below the target, so the realised FPR lands under 1% rather than on it:
the statistic is discrete and cannot hit an arbitrary rate exactly.

    python detect_report.py --in results/detect --out results/evals
"""

import argparse
import glob
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

# Display name -> key used by detect_eval, in table order.
ROWS = [
    ("No Distortion", "none"),
    ("Resample (8k)", "resample_8k"),
    ("Gaussian Noise", "gaussian_noise_20"),
    ("Median Filter", "median_filter"),
    ("Low-pass 4k", "low_pass_4k"),
    ("High-pass 500", "high_pass_500"),
    ("Reencode", "reencode"),
    ("Compression", "compression"),
    ("Noise Suppression", "noise_suppression"),
    ("Phone Call", "phone_call"),
]
METHODS = ["RT-SW", "AudioSeal", "WavMark", "Timbre", "SilentCipher"]
DATASETS = ["LibriSpeech-dev", "LJSpeech", "clone_xspeaker", "resynth_hifigan"]
TARGET_FPR = 0.01
Z = 1.959963984540054  # two-sided 95%


# ------------------------------------------------------------------ statistics


def wilson(k, n, z=Z):
    """Wilson score interval. Normal approximation collapses at p near 0 or 1,
    which is exactly where TPR and FPR sit here."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc_ci(pos, neg, n_boot=2000, seed=0):
    """AUC with a percentile bootstrap CI, resampling positives and negatives
    independently."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    if len(np.unique(s)) == 1:  # degenerate: every score identical
        return 0.5, 0.5, 0.5
    point = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        p = rng.choice(pos, len(pos), replace=True)
        q = rng.choice(neg, len(neg), replace=True)
        if len(np.unique(np.r_[p, q])) == 1:
            boot[b] = 0.5
            continue
        boot[b] = roc_auc_score(y, np.r_[p, q])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def calibrate(neg, n_bits, target=TARGET_FPR):
    """Smallest integer threshold whose FPR on `neg` is <= target."""
    for t in range(0, n_bits + 1):
        if float(np.mean(neg >= t)) <= target:
            return t
    return n_bits + 1


def cell(pos, neg, n_bits, thr, seed=0):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    n = len(pos)
    acc = pos / n_bits
    mean = float(acc.mean()) if n else float("nan")
    std = float(acc.std(ddof=1)) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    a, alo, ahi = auc_ci(pos, neg, seed=seed)
    ktp, ntp = int((pos >= thr).sum()), n
    kfp, nfp = int((neg >= thr).sum()), len(neg)
    tlo, thi = wilson(ktp, ntp)
    flo, fhi = wilson(kfp, nfp)
    return dict(
        n=n, acc=mean, acc_std=std, acc_sem=sem,
        acc_lo=mean - Z * sem if n > 1 else float("nan"),
        acc_hi=mean + Z * sem if n > 1 else float("nan"),
        auc=a, auc_lo=alo, auc_hi=ahi,
        tpr=ktp / ntp if ntp else float("nan"), tpr_lo=tlo, tpr_hi=thi,
        fpr=kfp / nfp if nfp else float("nan"), fpr_lo=flo, fpr_hi=fhi,
        thr=thr,
    )


# ------------------------------------------------------------------ loading


def load(indir):
    """{method: {dataset: {'pos'/'neg': {row: array}, 'n_bits': int}}}"""
    data = {}
    for f in sorted(glob.glob(os.path.join(indir, "*.npz"))):
        base = os.path.basename(f)[:-4]
        if base.startswith("_"):
            continue
        m = re.match(r"^(.*)_(%s)$" % "|".join(map(re.escape, DATASETS)), base)
        if not m:
            print(f"  skipping unrecognised {base}")
            continue
        meth, ds = m.group(1), m.group(2)
        d = np.load(f, allow_pickle=True)
        data.setdefault(meth, {})[ds] = dict(
            n_bits=int(d["n_bits"]), used=int(d["used"]),
            pos={k: d[f"pos_{k}"] for _, k in ROWS if f"pos_{k}" in d},
            neg={k: d[f"neg_{k}"] for _, k in ROWS if f"neg_{k}" in d},
        )
    return data


# ------------------------------------------------------------------ workbook

HDR = ["distortion", "n", "bit_acc", "acc_std", "acc_sem", "acc_ci_lo",
       "acc_ci_hi", "auc", "auc_ci_lo", "auc_ci_hi", "tpr", "tpr_ci_lo",
       "tpr_ci_hi", "fpr", "fpr_ci_lo", "fpr_ci_hi", "threshold_bits"]


def _sheet(ws, title):
    ws.title = title
    return ws


def write_xlsx(path, per, summary, thresholds, order):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    bold = Font(bold=True)

    ws = wb.active
    ws.title = "Summary"
    ws.append(["Averaged over the four evaluation corpora (unweighted mean of "
               "per-corpus values; CI columns are pooled over all clips)"])
    ws["A1"].font = bold
    ws.append([])
    for meth in order:
        ws.append([meth])
        ws.cell(ws.max_row, 1).font = bold
        ws.append(HDR)
        for c in range(1, len(HDR) + 1):
            ws.cell(ws.max_row, c).font = bold
        for label, key in ROWS:
            r = summary[meth][key]
            ws.append([label, r["n"]] + [round(r[k], 6) for k in HDR[2:]])
        ws.append([])

    for ds in order_datasets(per):
        ws = wb.create_sheet(ds[:31])
        ws.append([f"{ds}"])
        ws["A1"].font = bold
        ws.append([])
        for meth in order:
            if ds not in per.get(meth, {}):
                continue
            ws.append([meth])
            ws.cell(ws.max_row, 1).font = bold
            ws.append(HDR)
            for c in range(1, len(HDR) + 1):
                ws.cell(ws.max_row, c).font = bold
            for label, key in ROWS:
                r = per[meth][ds].get(key)
                if r is None:
                    continue
                ws.append([label, r["n"]] + [round(r[k], 6) for k in HDR[2:]])
            ws.append([])

    ws = wb.create_sheet("Thresholds")
    ws.append(["method", "payload_bits", "threshold_bits", "pooled_negatives",
               "realised_pooled_fpr", "target_fpr"])
    for c in range(1, 7):
        ws.cell(1, c).font = bold
    for meth in order:
        t = thresholds[meth]
        ws.append([meth, t["n_bits"], t["thr"], t["n_neg"],
                   round(t["fpr"], 6), TARGET_FPR])

    ws = wb.create_sheet("Raw")
    ws.append(["method", "dataset"] + HDR)
    for c in range(1, len(HDR) + 3):
        ws.cell(1, c).font = bold
    for meth in order:
        for ds in order_datasets(per):
            for label, key in ROWS:
                r = per.get(meth, {}).get(ds, {}).get(key)
                if r is None:
                    continue
                ws.append([meth, ds, label, r["n"]]
                          + [round(r[k], 6) for k in HDR[2:]])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wb.save(path)


def order_datasets(per):
    seen = [d for d in DATASETS if any(d in v for v in per.values())]
    return seen


# ------------------------------------------------------------------ LaTeX

TEX_HEAD = r"""\begin{table*}[t]
\centering
\caption{
Detection results under different audio distortions.
Acc. denotes detection accuracy, TPR/FPR denotes the true-positive
and false-positive rates, and AUC denotes the area under the ROC curve.
}
\label{tab:distortion_comparison}

\resizebox{\textwidth}{!}{
\begin{tabular}{l ccc ccc ccc ccc ccc}
\toprule
& \multicolumn{3}{c}{\textbf{RT-SW}}
& \multicolumn{3}{c}{\textbf{AudioSeal}}
& \multicolumn{3}{c}{\textbf{WavMark}}
& \multicolumn{3}{c}{\textbf{Timbre}}
& \multicolumn{3}{c}{\textbf{SilentCipher}} \\

\cmidrule(lr){2-4}
\cmidrule(lr){5-7}
\cmidrule(lr){8-10}
\cmidrule(lr){11-13}
\cmidrule(lr){14-16}

\textbf{Distortion}
& \textbf{Acc.} & \textbf{TPR/FPR} & \textbf{AUC}
& \textbf{Acc.} & \textbf{TPR/FPR} & \textbf{AUC}
& \textbf{Acc.} & \textbf{TPR/FPR} & \textbf{AUC}
& \textbf{Acc.} & \textbf{TPR/FPR} & \textbf{AUC}
& \textbf{Acc.} & \textbf{TPR/FPR} & \textbf{AUC} \\
\midrule
"""

TEX_TAIL = r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""


def fmt(v, nd=3):
    return "--" if v != v else f"{v:.{nd}f}"


def write_tex(path, summary):
    lines = []
    for label, key in ROWS:
        parts = [label]
        for meth in METHODS:
            r = summary.get(meth, {}).get(key)
            if r is None:
                parts.append("& -- & -- & --")
            else:
                parts.append(f"& {fmt(r['acc'])} & {fmt(r['tpr'])}/"
                             f"{fmt(r['fpr'])} & {fmt(r['auc'])}")
        lines.append("\n".join(parts) + r" \\")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(TEX_HEAD + "\n\n".join(lines) + TEX_TAIL)


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results/detect")
    ap.add_argument("--out", default="results/evals")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    data = load(args.indir)
    if not data:
        raise SystemExit(f"no .npz under {args.indir}")

    order = [m for m in METHODS if m in data] + \
            [m for m in sorted(data) if m not in METHODS]

    # One threshold per method, from every negative it produced.
    thresholds = {}
    for meth, byds in data.items():
        n_bits = next(iter(byds.values()))["n_bits"]
        pooled = np.concatenate([v["neg"][k] for v in byds.values()
                                 for _, k in ROWS if k in v["neg"]])
        thr = calibrate(pooled, n_bits)
        thresholds[meth] = dict(n_bits=n_bits, thr=thr, n_neg=len(pooled),
                                fpr=float(np.mean(pooled >= thr)))
        print(f"{meth:<13} payload {n_bits:>2} bits, threshold >= {thr} bits, "
              f"pooled FPR {thresholds[meth]['fpr']:.4f} "
              f"on {len(pooled)} negatives")

    per, summary = {}, {}
    for meth, byds in data.items():
        n_bits = thresholds[meth]["n_bits"]
        thr = thresholds[meth]["thr"]
        per[meth] = {}
        for ds, v in byds.items():
            per[meth][ds] = {
                k: cell(v["pos"][k], v["neg"][k], n_bits, thr)
                for _, k in ROWS if k in v["pos"]
            }
        # Averaged row: pooled clips give the CI, mean-of-corpora the headline.
        summary[meth] = {}
        for _, k in ROWS:
            dss = [ds for ds in byds if k in byds[ds]["pos"]]
            if not dss:
                continue
            pos = np.concatenate([byds[ds]["pos"][k] for ds in dss])
            neg = np.concatenate([byds[ds]["neg"][k] for ds in dss])
            r = cell(pos, neg, n_bits, thr)
            for f in ("acc", "auc", "tpr", "fpr"):
                r[f] = float(np.mean([per[meth][ds][k][f] for ds in dss]))
            summary[meth][k] = r

    xlsx = os.path.join(args.out, "detection_metrics.xlsx")
    tex = os.path.join(args.out, "distortion_comparison.tex")
    write_xlsx(xlsx, per, summary, thresholds, order)
    write_tex(tex, summary)

    print(f"\nwrote {xlsx}\nwrote {tex}\n")
    hdr = f"{'distortion':<20}" + "".join(f"{m:>26}" for m in order)
    print(hdr)
    print("-" * len(hdr))
    for label, key in ROWS:
        row = f"{label:<20}"
        for meth in order:
            r = summary.get(meth, {}).get(key)
            row += "  " + (f"{'--':>24}" if r is None else
                           f"{r['acc']:.3f} {r['tpr']:.3f}/{r['fpr']:.3f} "
                           f"{r['auc']:.3f}".rjust(24))
        print(row)
    missing = [(m, d) for m in order for d in DATASETS if d not in per.get(m, {})]
    if missing:
        print("\nstill missing: " + ", ".join(f"{m}/{d}" for m, d in missing))


if __name__ == "__main__":
    main()
