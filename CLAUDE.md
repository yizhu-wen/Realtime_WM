# Project notes

Real-time (streaming) neural speech watermarking. Fork of
[TimbreWatermarking](https://github.com/TimbreWatermarking/TimbreWatermarking) (NDSS 2024),
adapted so the encoder embeds with only a short lookahead.

`watermarking_model/train.py` is the only entry point. Everything tracked is
reachable from it — 11 `.py` files plus 3 configs (`move_flac_to_wav.py` is the
one exception: dataset prep, not imported).

## Environment

Not the system python. Use the `timbrewm` conda env:

```bash
/home/yizwen/.conda/envs/timbrewm/bin/python    # torch 2.7.1+cu128, py3.11
```

`wandb` and `silero-vad` were installed into it on 2026-09-04; they are missing
from older env snapshots. `requirements.txt` is pinned to what actually runs.

Dataset at `/data/yizwen/LibriSpeech_wav/` — train 28539 / val 2703 / test 2620 wav.
Clips shorter than `2s + delay_amt_second + future_amt_second` are filtered out
at dataset construction.

GPU: one RTX PRO 6000 (97 GB). **Do not run two trainings concurrently** — two
runs at batch 8 exhausted it and both died with `cuFFT error: CUFFT_INTERNAL_ERROR`,
which looks nothing like an OOM. Run sequentially.

## Running

```bash
cd watermarking_model
export WANDB_API_KEY=<key>     # required when wandb.enabled: true
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml
# or: ./shs/train.sh
```

A missing `WANDB_API_KEY` now fails in seconds with instructions, before the
~28k-file dataset scan. Set `wandb.enabled: false` to train without logging;
train and val metrics both print to stdout either way.

For a fast smoke test: symlink ~400 wavs into a scratch `train/ val/ test/`
layout, point `path.raw_path` at it, set `epoch: 15`, `wandb.enabled: false`.
That is ~50 steps/epoch and ~1 min/epoch.

## State as of 2026-09-04

Branch `from-0.25s`, pushed and in sync. Five commits this session
(`0a73fae..8e792bf`):

| commit | what |
| --- | --- |
| `2b77fca` | trainer bug fixes (below) |
| `588ce28` | removed code unrelated to the trainer; wandb key out of config |
| `63f55e4` | root README rewritten for this fork |
| `f5989e7` | removed 37 config keys the code never reads |
| `8e792bf` | epoch 60 → 50 |

### The bug that mattered

`lambda_a = lambda_m = train_config["optimize"]["lambda_a"]` appeared in the
train, val **and** test loops. It rebound the `main()`-local `lambda_m` from the
configured 10.0 down to 0.01. The val-loop copy runs at the end of every epoch,
so even a run that starts correct is poisoned from epoch 2 on. With the message
term that weak the model minimises the perceptual terms instead — SNR climbs to
~50 dB with a vanishing watermark while bit accuracy sits at chance.

Fixing only the training-loop copy is **not enough** and looks like the fix
failed; that is what the val-loop copy does. All three are gone now.

400 files, 20 epochs, same seed:

| | test bit acc | test SNR |
| --- | --- | --- |
| before | 0.48 / 0.49 (chance) | 49.7 dB |
| after | 0.79 / 0.91 | −2.0 dB |

Other fixes in `2b77fca`: discriminator was stepping on gradients contaminated
by the generator's `g_loss_adv` (`sum_loss.backward()` deposits them, and
`my_step(en_de_op, …)` only zeroes encoder/decoder — now `d_op.zero_grad()`
first); `adv: False` crashed on unguarded discriminator references; checkpoints
omitted the discriminator so adversarial runs could not resume; plus an
operator-precedence error in the encoder chunk guard and an `UnboundLocalError`
if `smooth_chunks`/`dilate_chunks` were set non-null.

## Last run: stopped by user after 5 epochs (2026-09-05)

Full-dataset 50-epoch run, killed at 3h31m by user request. Not a crash.

- wandb: https://wandb.ai/yizhuwenus-university-of-hawaii-system/real-time-voice-watermark/runs/wgr5miqz
- log: `watermarking_model/results/log/fulltrain_20260904_221352.log` (gitignored)
- checkpoint kept: `results/ckpt/pth/none-conv2_ep_5_2026-09-05_01_38_25.pth.tar`
  (encoder, decoder, discriminator, both optimizers; resumable)
- settings: 28539 train files, batch 8, `lambda_m: 10`, `lambda_b: 1`,
  delay/future 0.5s, `adv: True`, `distortion: true`

### What it showed

The message decoder learns very well at full scale, and far faster than on the
400-file subset — but SNR degrades monotonically. Validation:

| epoch | val bit acc | val SNR |
| --- | --- | --- |
| 1 | 0.815 / 0.868 | +0.59 dB |
| 2 | 0.877 / 0.899 | −1.51 dB |
| 3 | 0.890 / 0.925 | −2.12 dB |
| 4 | 0.921 / 0.955 | −3.19 dB |
| 5 | 0.929 / 0.969 | −3.46 dB |

Train acc hit 0.985 / 0.994 by epoch 5, and individual steps reached 1.000.

So accuracy is not the problem — imperceptibility is. The +2.3 dB reading at
step 1000 of epoch 1 was a transient, not a trend; SNR fell steadily after it.
The encoder is buying bit accuracy by making the watermark louder, and nothing
bounds that: `TFLoudnessRatio` is unbounded below and the waveform MSE term is
orders of magnitude smaller than the message term.

Corroborating: both discriminator losses collapsed to ~1e-3 by epoch 5, i.e.
the discriminator separates watermarked from cover audio trivially — consistent
with a clearly audible watermark. With `lambda_a: 0.01` the adversarial term is
far too weak to push back.

**Conclusion: the loss weights do need retuning, at full scale too.** Raise
`lambda_b` (and/or `lambda_a`) relative to `lambda_m: 10`, or add an explicit
SNR floor / watermark-amplitude penalty. Five epochs is ~3.5 h, which is a
usable calibration loop — accuracy is already >0.92 by then, so the question is
purely how much SNR can be bought back.

## Open items

1. **Retune the loss weights.** Confirmed necessary at full scale by the run
   above: val SNR fell from +0.6 dB to −3.5 dB over 5 epochs while accuracy
   climbed to 0.93/0.97. The message term dominates and nothing bounds the
   watermark's amplitude. Use ~5-epoch runs (~3.5 h) as the calibration loop.

2. **Rotate the wandb API key.** It was committed in `config/train.yaml` and is
   still in git history (removing it from the working tree does not retract it).
   All those commits were on GitHub before this work started. Rotate at
   wandb.ai/settings.

3. **`future_amt_second` is 0.5, but this branch is named `from-0.25s`.**
   `d4fcb67` set it to 0.25; `0e7c300 "tide up"` reverted it to 0.5 inside a
   cleanup commit, which looks unintentional. Decide which value this branch is
   supposed to be testing. Not touched by this session's commits.

## Things worth knowing about the code

- Encoder returns `(masked_y, zeros_right)`; the watermark is gated by a
  silero-VAD soft mask with a dynamic per-sample floor driven by frame RMS.
- Decoder returns two messages: `decoded[0]` from the distorted path
  (RIR + noise + bandpass 500–2000 Hz, hardcoded in `Decoder.__init__`, *not*
  from config) and `decoded[1]` from the clean identity path. Both share one
  extractor, so a bad distorted path drags the clean one down too.
- `offset_samples = 40480` is hardcoded in `train.py` — `((204+50)-1)*160`, tied
  to `audio_prefilling` and `delay_amt_second`. It does **not** track config
  changes; revisit it if you change those.
- `TFLoudnessRatio` is unbounded below — it keeps rewarding a quieter watermark
  with no floor. Relevant to item 1.
- Loss *values* are misleading for judging which term dominates; the waveform
  MSE is ~1e-6 while the message term is ~2. Compare **gradient norms** instead
  (with `lambda_m: 10` the message term is ~99.6% of the update).
