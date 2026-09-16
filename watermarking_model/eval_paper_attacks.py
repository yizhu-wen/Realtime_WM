"""RT-SW against the distortion suites of AudioSeal, Timbre and SilentCipher.

Small-scale by design: 200 LibriSpeech dev clips, RT-SW only. Same detection
protocol as detect_eval.py -- per clip a positive (watermarked, distorted,
scored against the embedded message) and a negative (the same clip
unwatermarked, distorted, scored against an independent random message) -- so
the numbers are comparable with the main table.

    python eval_paper_attacks.py --n_items 200
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detect_eval as de                                   # noqa: E402
from detect_report import (Z, calibrate, calibrate_randomized,  # noqa: E402
                           apply_randomized, auc_ci, wilson)
from distortions.paper_attacks import ORDER, PARAMS, PaperAttacks  # noqa: E402

PAPER = {**{k: "AudioSeal" for k in
            ["boost", "duck", "mp3_64", "mp3_128", "mp3_256", "aac_64",
             "aac_128", "aac_256", "encodec", "smoothing", "speed_up",
             "pink_noise"]},
         **{k: "Timbre" for k in
            ["median_5", "median_15", "median_25", "median_35",
             "crop_middle", "crop_end"]},
         **{k: "SilentCipher" for k in
            ["ogg", "time_jitter", "quant_16bit", "speech_mix_-15dB",
             "sample_suppress"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/ckpt/pth/"
                    "none-conv2_ep_20_2026-09-11_06_02_42.pth.tar")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/evals/paper_attacks")
    args = ap.parse_args()

    m = de.RTSWMethod(args.ckpt)

    # Speed-up shortens by 1.25x and the crops remove 10%, so a clip must be
    # long enough that every attacked version still clears the encoder's 3 s
    # minimum. 4 s covers the worst case (3 s x 1.25).
    clips = de.list_clips("LibriSpeech-dev", m.sr, int(4.0 * m.sr))
    np.random.default_rng(0).shuffle(clips)
    pool = [de.load(c, m.sr) for c in clips[args.n_items:args.n_items + 20]]
    clips = clips[: args.n_items]
    print(f"RT-SW / LibriSpeech-dev: {len(clips)} clips >= 4 s, "
          f"{len(ORDER)} distortions, {m.n_bits}-bit payload", flush=True)

    pa = PaperAttacks(mixer_pool=pool, seed=0)
    pos = {k: [] for k in ORDER}
    neg = {k: [] for k in ORDER}
    rng = np.random.default_rng(args.seed)
    used = 0

    for i, f in enumerate(clips):
        try:
            x = de.load(f, m.sr)
            y, msg = m.embed(x, rng)
            if y is None:
                continue
            nmsg = m.random_msg(rng)
            p_i, n_i = {}, {}
            for k in ORDER:                      # score the clip whole or not at all
                p_i[k] = m.decode(pa.apply(k, y), msg)
                n_i[k] = m.decode(pa.apply(k, x), nmsg)
            for k in ORDER:
                pos[k].append(p_i[k]); neg[k].append(n_i[k])
            used += 1
        except Exception as e:
            print(f"  clip {i} failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(clips)} ({used} used)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "scores.npz"), n_bits=m.n_bits, used=used,
             names=np.array(ORDER),
             **{f"pos_{k}": np.array(pos[k]) for k in ORDER},
             **{f"neg_{k}": np.array(neg[k]) for k in ORDER})

    # one operating point for the whole suite, from every negative it produced
    pooled = np.concatenate([neg[k] for k in ORDER])
    thr = calibrate(pooled, m.n_bits)
    rand = calibrate_randomized(pooled, m.n_bits)
    print(f"\nthreshold >= {thr}/{m.n_bits} bits (pooled FPR "
          f"{np.mean(pooled >= thr):.4f} on {len(pooled)} negatives); "
          f"randomised rule >= {rand[0]} or == {rand[0]-1} w.p. {rand[1]:.3f} "
          f"-> exactly 1.0%\n", flush=True)

    rows = []
    for k in ORDER:
        p, n = np.array(pos[k], float), np.array(neg[k], float)
        acc = p / m.n_bits
        sem = acc.std(ddof=1) / np.sqrt(len(acc))
        a, alo, ahi = auc_ci(p, n, n_boot=2000)
        rows.append(dict(
            name=k, paper=PAPER[k], note=PARAMS[k], n=len(acc),
            acc=acc.mean(), std=acc.std(ddof=1), ci=Z * sem,
            auc=a, auc_lo=alo, auc_hi=ahi,
            tpr=float(np.mean(p >= thr)), fpr=float(np.mean(n >= thr)),
            tpr_x=apply_randomized(p, *rand), fpr_x=apply_randomized(n, *rand)))
    return rows, args.out, m.n_bits, thr, rand, used


if __name__ == "__main__":
    rows, out, nb, thr, rand, used = main()
    import json
    json.dump(dict(rows=rows, n_bits=nb, thr=thr, gamma=rand[1], t=rand[0],
                   used=used), open(os.path.join(out, "summary.json"), "w"),
              indent=1)
    hi = [r for r in rows if r["acc"] > 0.80]
    w = f"{'distortion':<20}{'paper':<14}{'Acc':>8}{'+/-95%':>9}{'std':>7}{'AUC':>8}{'TPR@1%':>9}{'FPR':>8}"
    print(w); print("-" * len(w))
    for r in rows:
        mark = " *" if r["acc"] > 0.80 else "  "
        print(f"{r['name']:<20}{r['paper']:<14}{r['acc']:>8.4f}{r['ci']:>9.4f}"
              f"{r['std']:>7.3f}{r['auc']:>8.4f}{r['tpr_x']:>9.3f}{r['fpr_x']:>8.4f}{mark}")
    print("-" * len(w))
    print(f"* Acc > 80%: {len(hi)} of {len(rows)} -- "
          + ", ".join(r["name"] for r in hi))
