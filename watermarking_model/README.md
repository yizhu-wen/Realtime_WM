# Real-time speech watermarking

Streaming-capable neural speech watermarking. An encoder embeds an `n`-bit message
into speech using only a short lookahead, and a decoder recovers it from the
watermarked audio after transmission-style distortion.

Built on the watermarking model from
[TimbreWatermarking](https://github.com/TimbreWatermarking/TimbreWatermarking) (NDSS 2024).

## Layout

```
watermarking_model/
├── train.py                    # the trainer (only entry point)
├── config/
│   ├── process.yaml            # audio + STFT settings
│   ├── model.yaml              # encoder/decoder architecture
│   └── train.yaml              # paths, loss weights, schedule, wandb
├── model/
│   ├── conv2_mel_modules.py    # Encoder / Decoder / Discriminator
│   ├── blocks.py               # conv + FC building blocks
│   └── loss.py                 # MSE + TF-loudness losses
├── distortions/frequency.py    # differentiable STFT / iSTFT
├── dataset/
│   ├── data.py                 # WavDataset + collate_fn
│   ├── sh.sh                   # download LibriSpeech
│   └── move_flac_to_wav.py     # flac -> wav, into the layout below
└── utils/                      # optimizer step + checkpoint saving
```

## Setup

```bash
conda create -n timbrewm python=3.11
conda activate timbrewm
# install torch/torchaudio matched to your CUDA version first, then:
pip install -r requirements.txt
```

## Dataset

```bash
cd dataset && sh ./sh.sh
```

This downloads LibriSpeech and converts it to the layout the trainer expects.
Point `path.raw_path` in `config/train.yaml` at the resulting directory:

```
LibriSpeech_wav
├── train/    103-1240-0000.wav, ...
├── val/      1272-128104-0000.wav, ...
└── test/     1089-134686-0000.wav, ...
```

Clips shorter than `2s + delay_amt_second + future_amt_second` are filtered out
automatically at dataset construction.

## Training

```bash
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml
```

Each epoch logs train, then validation metrics (`wav_loss`, `msg_loss`,
`tfloudness_loss`, bit accuracy for both decoder heads, SNR, and the two
discriminator losses); a held-out test pass runs once at the end.
Checkpoints land in `path.ckpt` every `iter.save_circle` epochs and contain the
encoder, decoder, discriminator, and both optimizer states.

### Knobs worth knowing

| Config key | Meaning |
| --- | --- |
| `watermark.length` | message length in bits |
| `watermark.delay_amt_second` | hop between successive watermark chunks |
| `watermark.future_amt_second` | lookahead the encoder is allowed |
| `optimize.lambda_e / lambda_m / lambda_b / lambda_a` | weights for waveform, message, loudness, adversarial loss |
| `optimize.distortion` | apply RIR + noise + bandpass before decoding |
| `adv` | enable the discriminator |
| `iter.data_divider` | train on `1/N` of the data (quick runs) |

### wandb

`wandb.enabled: true` in `config/train.yaml` requires `WANDB_API_KEY` in the
environment. Set it once:

```bash
echo 'export WANDB_API_KEY=<your key>' >> ~/.bashrc && source ~/.bashrc
```

For batch jobs, export it in the submission script instead — a compute node may
not read your shell profile. The key is never read from the config file; do not
commit one there. If the variable is missing the run stops immediately with
instructions, rather than failing partway in.

To train without logging, set `wandb.enabled: false`.
