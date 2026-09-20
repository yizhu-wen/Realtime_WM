# Real-Time Speech Watermarking for Live Voice Communication: Synthetic Speech Provenance and Post-Hoc Recording Attribution

A Real-Time Speech Watermarking (RT-SW). The encoder embeds with **0.5 s of
lookahead** and is causal by construction, so it can run on live audio; the
baselines it is compared against all read the whole utterance offline.

## Result

RT-SW matches offline watermarks on standard distortions and is the only method
that survives a telephony channel (RIR + noise + 300–3400 Hz band):

| | RT-SW (ours) | AudioSeal | WavMark | Timbre | SilentCipher |
| --- | --- | --- | --- | --- | --- |
| Clean accuracy | 0.976 | 1.000 | 1.000 | 1.000 | 0.999 |
| **Telephony AUC** | **0.959** | 0.582 | 0.500 | 0.908 | 0.498 |
| SNR (dB) | 40.2 | 27.3 | 36.6 | 27.5 | 49.0 |
| PESQ | 4.16 | 4.43 | 4.15 | 3.70 | 4.59 |

Full tables: `watermarking_model/results/evals/`.

## Install

```bash
conda create -n rtsw python=3.11 && conda activate rtsw
pip install -r watermarking_model/requirements.txt
```

Needs a CUDA GPU and `ffmpeg` on `PATH`.

## Data

```bash
bash scripts/download_data.sh ./data          # LibriSpeech + LJSpeech, ~30 GB
```

Two synthetic corpora are used for evaluation only. Each takes a GPU and extra
dependencies, and both are optional:

```bash
pip install speechbrain
python scripts/make_resynth_hifigan.py --librispeech data/LibriSpeech_wav \
    --out data/resynth_hifigan_wav                        # ~7 min

pip install f5-tts
python scripts/make_clone_xspeaker.py --librispeech data/LibriSpeech_wav \
    --transcripts data/LibriSpeech --out data/clone_xspeaker_wav   # ~5.5 h
```

`resynth_hifigan` is vocoder analysis–resynthesis: the mel is measured from the
real recording, only the waveform is neural. `clone_xspeaker` is fully
synthetic cross-speaker voice cloning. Point the evaluators at them by editing
`DATASETS` in `watermarking_model/detect_eval.py`.

## Train

Set `path.raw_path` in `watermarking_model/config/train.yaml` to your
`LibriSpeech_wav`, then:

```bash
cd watermarking_model
export WANDB_API_KEY=<key>        # or set wandb.enabled: false
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml
```

20 epochs on one GPU, ~10 h. Checkpoints land in `results/ckpt/`.

## Evaluate

The shipped model is `watermarking_model/checkpoints/rtsw_ep20.pth.tar`.

```bash
bash scripts/run_evaluation.sh                        # RT-SW, both suites
METHODS="RTSW AudioSeal WavMark" bash scripts/run_evaluation.sh
```

Writes per-utterance scores, then an Excel workbook (with std and 95 % CIs) and
LaTeX tables to `watermarking_model/results/evals/`. Runs checkpoint every 50
clips and resume, so interrupting is cheap.

Quick bit-accuracy check on one checkpoint:

```bash
cd watermarking_model && python evaluate.py --ckpt checkpoints/rtsw_ep20.pth.tar --n_items 200
```

## How it is measured

Per clip and per distortion, a **positive** is `decode(distort(embed(x)))`
scored against the embedded message, and a **negative** is `decode(distort(x))`
on the same clip, unwatermarked, scored against an independent random message.
The statistic is the bit-match count, so methods with different payloads
(10–40 bits) stay comparable.

TPR/FPR are reported at a threshold calibrated on each method's pooled
negatives to a **1 % false-positive rate**. The statistic is an integer, so a
deterministic threshold cannot hit 1 % exactly and holds a 10-bit method to a
rate up to 10× stricter than a 40-bit one; a randomised decision at the
boundary fixes that. Both variants are written — use
`distortion_comparison_exact1pct.tex`.

Evaluation runs at **batch size 1** deliberately: batching pads to the longest
item and the encoder writes a watermark into that padding, worth ~7–9 accuracy
points that do not exist at inference.

## Baselines (optional)

Comparing against AudioSeal, WavMark, Timbre and SilentCipher needs their
packages, weights, and a wrapper exposing `embed(x, rng)` / `decode(x, msg)`
per method. `detect_eval.py` expects it at `/data/yizwen/wm_shift_exp/methods.py`;
edit that path for your setup. RT-SW alone needs none of this.

## Layout

```
scripts/            data download and synthetic-corpus generation
watermarking_model/
  train.py          training entry point
  evaluate.py       bit accuracy for one checkpoint
  detect_eval.py    per-utterance detection scores  (--suite dl|paper)
  imperceptibility.py   SNR and PESQ
  detect_report.py  scores -> Excel + LaTeX
  merge_tables.py   combine both distortion suites into one table
  config/ model/ dataset/ distortions/ utils/
  checkpoints/      the released model
  results/evals/    tables
```

Base model architecture: Fork of [TimbreWatermarking](https://github.com/TimbreWatermarking/TimbreWatermarking) (NDSS 2024).

## License

See `LICENSE`.
