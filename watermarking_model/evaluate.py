"""Evaluate a watermarking checkpoint: bit accuracy per dataset per distortion.

Runs at batch size 1 by default. Batched evaluation pads short items to the
batch maximum, and the encoder writes a watermark into that padding which the
decoder's mean over the time axis then reads back -- worth ~7-9 accuracy points
that do not exist at inference on a single utterance.

    python evaluate.py --ckpt results/ckpt/pth/none-conv2_ep_50_....pth.tar
"""

import argparse
import glob
import os
import sys
import warnings

import torch
import torchaudio
import yaml
from rich.progress import track
from torch.nn.functional import mse_loss

from dataset.data import WavDataset
from distortions.dl import distortion
from model.conv2_mel_modules import Decoder, Encoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# name -> (attack index in distortions.dl, ratio)
DISTORTIONS = [
    ("none", 0, 0),
    ("resample_8k", 8, 0),
    ("gaussian_noise_20", 9, 20),
    ("median_filter", 13, 3),
    ("low_pass_2k", 31, 0),
    ("low_pass_4k", 32, 0),
    ("high_pass_500", 15, 0),
    ("reencode", 27, 0),
    ("compression", 29, 0),
    ("noise_suppression", 28, 0),
    ("phone_call", 30, 0),
    ("phone_call_legacy", 33, 0),
    ("phone_call_trainmatch", 34, 0),
]


class FlatWavDataset(torch.utils.data.Dataset):
    """Every .wav under a directory, resampled to the model's rate.

    LJSpeech ships at 22.05 kHz and is split across the root and a wavs/
    subdirectory, so this recurses and resamples rather than assuming the
    LibriSpeech train/val/test layout WavDataset expects.
    """

    def __init__(self, root, target_sr, min_samples, limit=None):
        self.target_sr = target_sr
        files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
        self.files = []
        for f in files:
            try:
                info = torchaudio.info(f)
            except Exception:
                continue
            # length after resampling must still clear the encoder's minimum
            if info.num_frames / info.sample_rate * target_sr >= min_samples:
                self.files.append(f)
            if limit and len(self.files) >= limit:
                break
        self._resamplers = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        wav, sr = torchaudio.load(self.files[i])
        wav = wav[0]
        if sr != self.target_sr:
            if sr not in self._resamplers:
                self._resamplers[sr] = torchaudio.transforms.Resample(
                    sr, self.target_sr
                )
            wav = self._resamplers[sr](wav)
        return wav


class LibriTestDataset(torch.utils.data.Dataset):
    """Thin wrapper so the LibriSpeech split yields bare waveforms too."""

    def __init__(self, process_config, train_config, flag, limit=None):
        self.inner = WavDataset(
            process_config=process_config, train_config=train_config, flag=flag
        )
        self.n = len(self.inner) if limit is None else min(limit, len(self.inner))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.inner[i]["matrix"]


def evaluate(encoder, decoder, dataset, msg_length, sample_rate, distorter, n_items):
    """Return {distortion_name: (bit_accuracy, n_utterances)} plus mean SNR."""
    # The decoder applies its own training distortion internally; disable it so
    # the only channel applied is the one under test.
    was_distorting = decoder.distortion
    decoder.distortion = False

    correct = {name: 0 for name, _, _ in DISTORTIONS}
    total = {name: 0 for name, _, _ in DISTORTIONS}
    snr_sum, snr_n = 0.0, 0
    used = 0

    n = min(n_items, len(dataset))
    for i in track(
        range(n), description="  evaluating", disable=not sys.stdout.isatty()
    ):
        x = dataset[i].unsqueeze(0).to(device)  # [1, T], no padding
        msg = (
            torch.randint(0, 2, (1, 1, msg_length), device=device).float() * 2
        ) - 1
        with torch.no_grad():
            out = encoder(x, msg, 1)
            if out is None:  # too short for even one chunk
                continue
            wm, _ = out
            y = x + wm
            used += 1
            z = torch.zeros_like(x)
            snr_sum += (10 * torch.log10(mse_loss(x, z) / mse_loss(x, y))).item()
            snr_n += 1

            for name, idx, ratio in DISTORTIONS:
                try:
                    y_d = distorter(y.unsqueeze(1), attack_choice=idx, ratio=ratio)
                    if y_d.dim() == 3:
                        y_d = y_d.squeeze(1)
                    # some channels (resampling, codecs) change length slightly
                    if y_d.shape[-1] != y.shape[-1]:
                        m = min(y_d.shape[-1], y.shape[-1])
                        y_d = y_d[..., :m]
                    decoded, _ = decoder(y_d, 1)
                except Exception as e:
                    print(f"    [{name}] failed on item {i}: {type(e).__name__}: {e}")
                    continue
                correct[name] += (decoded >= 0).eq(msg >= 0).sum().item()
                total[name] += msg.numel()

    decoder.distortion = was_distorting
    acc = {
        n: (correct[n] / total[n] if total[n] else float("nan"), total[n] // msg_length)
        for n, _, _ in DISTORTIONS
    }
    return acc, (snr_sum / snr_n if snr_n else float("nan")), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-p", "--process_config", default="./config/process.yaml")
    ap.add_argument("-m", "--model_config", default="./config/model.yaml")
    ap.add_argument("-t", "--train_config", default="./config/train.yaml")
    ap.add_argument("--n_items", type=int, default=200, help="utterances per dataset")
    ap.add_argument(
        "--libri_splits",
        default="val,test",
        help="LibriSpeech splits to score. 'val' is dev-clean in this layout.",
    )
    ap.add_argument(
        "--ljspeech", default="/data/yizwen/LJSpeech-1.1_wav", help="'' to skip"
    )
    args = ap.parse_args()

    process_config = yaml.load(open(args.process_config), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config), Loader=yaml.FullLoader)

    msg_length = train_config["watermark"]["length"]
    sr = process_config["audio"]["or_sample_rate"]

    encoder = Encoder(process_config, model_config, train_config, msg_length).to(device)
    decoder = Decoder(process_config, model_config, train_config, msg_length).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()
    print(f"checkpoint: {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    distorter = distortion(process_config).to(device)
    min_samples = int(
        2 * sr
        + train_config["watermark"]["delay_amt_second"] * sr
        + train_config["watermark"]["future_amt_second"] * sr
    )

    # In this dataset layout dev-clean lives in val/ (see dataset/move_flac_to_wav.py)
    split_label = {"val": "LibriSpeech-dev", "test": "LibriSpeech-test"}
    datasets = {}
    for sp in [x.strip() for x in args.libri_splits.split(",") if x.strip()]:
        datasets[split_label.get(sp, f"LibriSpeech-{sp}")] = LibriTestDataset(
            process_config, train_config, sp, limit=args.n_items
        )
    if args.ljspeech:
        datasets["LJSpeech"] = FlatWavDataset(
            args.ljspeech, sr, min_samples, limit=args.n_items
        )

    results = {}
    for name, ds in datasets.items():
        print(f"\n{name}: {len(ds)} utterances available")
        acc, snr, used = evaluate(
            encoder, decoder, ds, msg_length, sr, distorter, args.n_items
        )
        results[name] = (acc, snr, used)

    print("\n" + "=" * 74)
    print("Bit accuracy by distortion (batch size 1, no padding)")
    print("=" * 74)
    header = f"{'distortion':<20}" + "".join(f"{n:>22}" for n in results)
    print(header)
    print("-" * len(header))
    for dname, _, _ in DISTORTIONS:
        row = f"{dname:<20}"
        for _, (acc, _, _) in results.items():
            a, n = acc[dname]
            row += f"{a:>16.4f} ({n:>3})"
        print(row)
    print("-" * len(header))
    row = f"{'SNR (dB)':<20}"
    for _, (_, snr, _) in results.items():
        row += f"{snr:>22.2f}"
    print(row)


if __name__ == "__main__":
    main()
