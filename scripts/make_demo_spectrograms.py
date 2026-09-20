"""Spectrograms for the project page: one per demo clip, plus the watermark
residual panels that mirror Fig. 3 of the paper."""
import glob, os
import numpy as np, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUD = os.path.join(REPO, "docs/assets/audio")
OUT = os.path.join(REPO, "docs/assets/spec")
os.makedirs(OUT, exist_ok=True)
SR, NFFT, HOP = 16000, 512, 128
ORDER = [("clean", "Clean"), ("rtsw", "RT-SW (ours)"), ("audioseal", "AudioSeal"),
         ("wavmark", "WavMark"), ("timbre", "Timbre"),
         ("silentcipher", "SilentCipher")]


def spec_db(x):
    f = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP] * np.hanning(NFFT),
        axis=-1)).T
    return 20*np.log10(f + 1e-10)


def panel(mat, path, title, vmin, vmax, dur, cmap="magma"):
    fig, ax = plt.subplots(figsize=(4.2, 2.0), dpi=150)
    ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              extent=[0, dur, 0, SR/2000])
    ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("kHz", fontsize=8)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


for tag, label in ORDER:
    f = os.path.join(AUD, f"{tag}.wav")
    if not os.path.exists(f):
        continue
    x, _ = sf.read(f, dtype="float32")
    dur = len(x)/SR
    panel(spec_db(x), os.path.join(OUT, f"{tag}.png"), label, -80, 0, dur)
    print(f"  {tag}.png")
    r = os.path.join(AUD, f"{tag}_residual.npy")
    if os.path.exists(r):
        # watermark only, on the fixed -80..-40 dB scale Fig. 3 uses
        panel(spec_db(np.load(r)), os.path.join(OUT, f"{tag}_residual.png"),
              f"{label} — watermark", -80, -40, dur, cmap="magma")
        print(f"  {tag}_residual.png")
