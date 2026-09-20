"""Watermark one LibriSpeech clip with every method, for the project page.

Writes, per method, the watermarked wav and the watermark residual
(watermarked - clean) that Fig. 3 of the paper plots.

    DATA_ROOT=/data/yizwen python scripts/make_demo_audio.py --method RTSW
    DATA_ROOT=/data/yizwen python scripts/make_demo_audio.py --method AudioSeal
"""
import argparse, os, sys
import numpy as np, soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "watermarking_model"))

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True)
ap.add_argument("--clip", default="")
ap.add_argument("--ckpt", default=os.path.join(REPO, "watermarking_model/checkpoints/rtsw_ep20.pth.tar"))
ap.add_argument("--out", default=os.path.join(REPO, "docs/assets/audio"))
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

sys.argv = ["x"]
import detect_eval as de

m = de.RTSWMethod(a.ckpt) if a.method == "RTSW" else de.BaselineMethod(a.method)
clip = a.clip or sorted(__import__("glob").glob(
    os.path.join(de.DATASETS["LibriSpeech-dev"], "*.wav")))[0]

x = de.load(clip, m.sr)
y, msg = m.embed(x, np.random.default_rng(a.seed))
y = np.asarray(y, dtype=np.float32).reshape(-1)
k = min(len(x), len(y)); x, y = x[:k], y[:k]

os.makedirs(a.out, exist_ok=True)
tag = a.method.lower()
# everything is written at 16 kHz so the page's clips are directly comparable
import scipy.signal as ss
to16 = lambda v: v if m.sr == 16000 else ss.resample_poly(v, 16000, m.sr).astype(np.float32)
if not os.path.exists(os.path.join(a.out, "clean.wav")):
    sf.write(os.path.join(a.out, "clean.wav"), to16(x), 16000, subtype="PCM_16")
sf.write(os.path.join(a.out, f"{tag}.wav"), to16(y), 16000, subtype="PCM_16")
np.save(os.path.join(a.out, f"{tag}_residual.npy"), to16(y - x))

snr = 10*np.log10(np.mean(x.astype(np.float64)**2) / np.mean((y-x).astype(np.float64)**2))
bits = m.decode(y, msg)
print(f"{a.method:<13} clip={os.path.basename(clip)} sr={m.sr} "
      f"SNR={snr:.2f} dB  recovered {bits}/{m.n_bits} bits")
