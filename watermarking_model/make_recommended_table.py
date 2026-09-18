import sys, numpy as np
sys.path.insert(0, "/home/yizwen/research/Realtime_WM/watermarking_model"); sys.argv = ["x"]
import merge_tables as mt

KEEP = [("No Distortion","none"), ("Median Filter","median_filter"),
        ("Low-pass 4k","low_pass_4k"), ("Noise Suppression","noise_suppression"),
        ("AAC 64 kbps","aac_64"), ("Sample Suppression","sample_suppress"),
        ("Telephony Distortion","phone_call")]

rows_all, data, thr, payload = mt.main()
def cell(m,k):
    p,n = data[m][k]; t,g,nb = thr[m]
    rej = lambda x: float(np.mean(x>=t)+g*np.mean(x==t-1))
    return p.mean()/nb, rej(p), rej(n), mt.auc_exact(p,n,nb)

HEAD = r"""% preamble: \newcommand{\aux}[1]{{\scriptsize{\textcolor{gray}{#1}}}}
\begin{table*}[t]
    \centering
    \vspace{-10pt}
    \caption{
        Decoding evaluation results under different audio distortions.
        RT-SW is the only streaming method: it embeds with 0.5\,s of lookahead,
        while every baseline reads the whole utterance offline.
    }
    \vspace{-10pt}
    \label{tab:wm_robustness}
    \resizebox{1.0\linewidth}{!}{
        \begin{tabular}{l *{2}{l} *{2}{l} *{2}{l} *{2}{l} *{2}{l}}
        \toprule
        & \multicolumn{2}{l}{\textbf{RT-SW (Ours)}}
        & \multicolumn{2}{l}{\textbf{AudioSeal}}
        & \multicolumn{2}{l}{\textbf{WavMark}}
        & \multicolumn{2}{l}{\textbf{Timbre}}
        & \multicolumn{2}{l}{\textbf{SilentCipher}} \\
        & \multicolumn{2}{l}{\aux{streaming, 0.5\,s lookahead}}
        & \multicolumn{2}{l}{\aux{offline}} & \multicolumn{2}{l}{\aux{offline}}
        & \multicolumn{2}{l}{\aux{offline}} & \multicolumn{2}{l}{\aux{offline}} \\
        \cmidrule(rr){2-3} \cmidrule(rr){4-5} \cmidrule(rr){6-7} \cmidrule(rr){8-9} \cmidrule(rr){10-11}
        \multicolumn{1}{c}{Distortion}
        & Acc. \aux{TPR/FPR} & AUC & Acc. \aux{TPR/FPR} & AUC
        & Acc. \aux{TPR/FPR} & AUC & Acc. \aux{TPR/FPR} & AUC
        & Acc. \aux{TPR/FPR} & AUC \\
        \cmidrule(rr){1-1} \cmidrule(rr){2-3} \cmidrule(rr){4-5} \cmidrule(rr){6-7} \cmidrule(rr){8-9} \cmidrule(rr){10-11}
"""
TAIL = r"""        \bottomrule
        \end{tabular}
    }
    \vspace{-0.2cm}
\end{table*}
"""
body = []
for label, k in KEEP:
    cells = []
    for m in mt.METHODS:
        a, tp, fp, au = cell(m, k)
        # bold only the telephony row, where the contribution lies; bolding
        # per-row winners elsewhere hands the eye to whichever baseline is
        # nominally ahead on conditions that separate nothing.
        b = r"\bf " if k == "phone_call" and m == "RT-SW" else ""
        cells.append(f"{b}{a:.3f} \\aux{{{tp:.3f}/{fp:.3f}}} & {b}{au:.3f}")
    body.append(f"        {label} & " + " & ".join(cells) + r" \\")
out = "results/evals/distortion_comparison_recommended.tex"
open(out, "w").write(HEAD + "\n".join(body) + "\n" + TAIL)
print(f"wrote {out}  ({len(KEEP)} rows)")
