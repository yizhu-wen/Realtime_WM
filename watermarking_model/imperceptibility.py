"""Imperceptibility of the watermark: SNR and PESQ, per utterance.

Measured on the clean/watermarked pair before any channel, so this is
independent of the detection protocol -- no distortion, no decoding, no
threshold. One encoder pass per clip.

    python imperceptibility.py --method RTSW --dataset LJSpeech --out x.npz

SNR is 10*log10(mean(x^2) / mean((y-x)^2)): speech power over watermark power,
the same definition evaluate.py reports, so the numbers are comparable with the
training-time figures.

PESQ is ITU-T P.862 wideband, which is only defined at 16 kHz. SilentCipher
works at 44.1 kHz, so both signals are resampled to 16 kHz for PESQ alone; SNR
stays at the method's native rate.
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_eval import (  # noqa: E402
    DATASETS, BaselineMethod, RTSWMethod, list_clips, load, method_sample_rate,
)

PESQ_SR = 16000


def snr_db(x, y):
    """Speech power over watermark power."""
    d = y - x
    num = float(np.mean(x.astype(np.float64) ** 2))
    den = float(np.mean(d.astype(np.float64) ** 2))
    if den <= 0:
        return float("inf")
    if num <= 0:
        return float("nan")
    return 10.0 * np.log10(num / den)


def to_pesq_rate(x, sr):
    if sr == PESQ_SR:
        return x
    import scipy.signal as ss

    return ss.resample_poly(x, PESQ_SR, sr).astype(np.float32)


def save(out, method, dataset, sr, snr, pesq_, used, done):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, method=method, dataset=dataset, sr=sr, used=used,
             complete=bool(done), snr=np.array(snr, dtype=np.float64),
             pesq=np.array(pesq_, dtype=np.float64))
    os.replace(tmp, out)
    if done:
        print(f"wrote {out}  ({used} clips)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["RTSW", "Timbre", "AudioSeal", "WavMark",
                             "SilentCipher"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--n_items", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import yaml
    from pesq import pesq as pesq_fn

    REPO = os.path.dirname(os.path.abspath(__file__))
    sr = method_sample_rate(args.method)
    m = RTSWMethod(args.ckpt) if args.method == "RTSW" \
        else BaselineMethod(args.method)
    assert m.sr == sr, (m.sr, sr)

    # Same duration filter and same shuffled order as detect_eval, so the two
    # measurements describe the same clips in the same sequence.
    tc = yaml.safe_load(open(os.path.join(REPO, "config/train.yaml")))
    min_s = 2 + tc["watermark"]["delay_amt_second"] \
        + tc["watermark"]["future_amt_second"]
    clips = list_clips(args.dataset, m.sr, int(min_s * m.sr))
    np.random.default_rng(0).shuffle(clips)
    if args.n_items:
        clips = clips[: args.n_items]
    print(f"{m.name} / {args.dataset}: {len(clips)} clips, sr {m.sr}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    snr, pq, used, failed = [], [], 0, 0
    for i, f in enumerate(clips):
        try:
            x = load(f, m.sr)
            y, _ = m.embed(x, rng)
            if y is None:
                continue
            k = min(len(x), len(y))
            x, y = x[:k], y[:k]
            snr.append(snr_db(x, y))
            try:
                pq.append(float(pesq_fn(PESQ_SR, to_pesq_rate(x, m.sr),
                                        to_pesq_rate(y, m.sr), "wb")))
            except Exception:
                # PESQ rejects clips with no detectable speech; keep the SNR
                # and record PESQ as missing rather than dropping the clip.
                pq.append(float("nan"))
            used += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  clip {i} failed: {type(e).__name__}: {str(e)[:90]}",
                      flush=True)
        if (i + 1) % 200 == 0:
            s = np.array(snr)
            p = np.array(pq)
            print(f"  {i+1}/{len(clips)} ({used} used)  "
                  f"SNR {np.nanmean(s):.2f} dB  PESQ {np.nanmean(p):.3f}",
                  flush=True)
            save(args.out, m.name, args.dataset, m.sr, snr, pq, used, False)

    save(args.out, m.name, args.dataset, m.sr, snr, pq, used, True)
    s, p = np.array(snr), np.array(pq)
    print(f"{m.name}/{args.dataset}: SNR {np.nanmean(s):.2f} +/- "
          f"{np.nanstd(s, ddof=1):.2f} dB, PESQ {np.nanmean(p):.3f} +/- "
          f"{np.nanstd(p, ddof=1):.3f}  ({used} clips, {failed} failed)",
          flush=True)


if __name__ == "__main__":
    main()
