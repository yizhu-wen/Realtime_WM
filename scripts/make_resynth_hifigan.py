"""Vocoder-resynthesised speech: real LibriSpeech waveforms reconstructed from
their own mel-spectrograms by HiFi-GAN.

Reproduces the re-synthesis condition of AudioSeal §5.1. The mel is *measured
from the real recording*; only the waveform reconstruction is neural, so
content, speaker, prosody and timing are all genuine. The accurate term is
analysis-resynthesis, not synthetic speech -- say so if you group it with fully
generated audio.

    python scripts/make_resynth_hifigan.py --librispeech data/LibriSpeech_wav \
        --out data/resynth_hifigan_wav --n 2000

Needs: pip install speechbrain
"""
import argparse, glob, json, os, random

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000
# front-end matched to the speechbrain/tts-hifigan-libritts-16kHz checkpoint
MEL = dict(sample_rate=SR, n_fft=1024, hop_length=256, win_length=1024,
           n_mels=80, f_min=0.0, f_max=8000.0, power=1.0,
           norm="slaney", mel_scale="slaney")
PAD = 1280          # HifiganGenerator.inference pads the mel; strip the offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True,
                    help="root with val/ and test/ of 16 kHz wavs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from speechbrain.inference.vocoders import HIFIGAN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    voc = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                               savedir="pretrained/hifigan", run_opts={"device": dev})
    mel_fn = torchaudio.transforms.MelSpectrogram(**MEL).to(dev)

    # dev-clean + test-clean, matching the published set
    files = sorted(sum((glob.glob(os.path.join(a.librispeech, s, "*.wav"))
                        for s in ("val", "test")), []))
    keep = [f for f in files
            if a.min_sec <= sf.info(f).frames / SR <= a.max_sec]
    random.Random(a.seed).shuffle(keep)
    keep = sorted(keep[: a.n])
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(keep)} clips -> {a.out}")

    rows = []
    for i, f in enumerate(keep):
        x, _ = sf.read(f, dtype="float32")
        t = torch.from_numpy(x)[None].to(dev)
        with torch.no_grad():
            m = mel_fn(t)
            m = torch.log(torch.clamp(m, min=1e-5))       # log compression
            y = voc.decode_batch(m)[0, 0].cpu().numpy()
        y = y[PAD: PAD + len(x)]                          # sample-align to source
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        uid = os.path.splitext(os.path.basename(f))[0]
        sf.write(os.path.join(a.out, uid + ".wav"), y, SR, subtype="PCM_16")
        rows.append(dict(id=uid, speaker=uid.split("-")[0], label="synthetic",
                         generator="hifigan:speechbrain-libritts-16kHz",
                         mode="resynth", ratio=1.0, path=f"{uid}.wav",
                         real_path=f, sample_rate=SR, num_samples=len(y),
                         duration=round(len(y) / SR, 4)))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(keep)}", flush=True)

    with open(os.path.join(a.out, "metadata.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} clips + metadata.jsonl")


if __name__ == "__main__":
    main()
