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

## Last run: completed but collapsed (2026-09-08)

50/50 epochs, 31.8 h, zero errors -- and no usable model. Run `oljqa8wp`,
`novad_50ep_lmeff0.01_band300-3400_causalRIR`.

The effective message weight of 0.01 removed any incentive to embed. The
encoder drove the watermark to numerical zero and the decoder learned to output
a constant 0:

- watermark RMS **1.03e-08**, max |wm| 3.08e-07. The 16-bit PCM step is
  3.05e-05, so the watermark is ~100x below the quantisation floor -- it cannot
  survive being written to a wav file.
- decoder output is exactly `[0, -0, -0, 0, ...]` for every bit
- `msg_loss` pinned at exactly 2.00000000 = MSE(msg, 0) x 2
- val SNR 131.8 dB; accuracy 0.49-0.51 under every distortion, both corpora

**It was diagnosable after epoch 1** (40 min): SNR 70.6 dB, acc 0.4996,
msg_loss 2.00020903. Nothing after that changed the outcome. Watch epoch 1 of
any weighting change before letting a run go 32 h.

### The two runs bracket the weighting

| effective lambda_m | outcome |
| --- | --- |
| 0.01 (this run) | watermark -> 0, SNR +131 dB, accuracy at chance |
| 10 (`b6hzmqft`) | message learned (0.70 honest / 0.97 padded), SNR -8.9 dB |

So a usable value is strictly between. Note that
`lambda_a = lambda_m = train_config["optimize"]["lambda_a"]` makes this
impossible to sweep: it ties the message weight to the adversarial weight, so
`lambda_m` cannot be varied from the config at all, and changing `lambda_a`
moves both. Removing that line and setting `lambda_m` in the config is what
makes a sweep possible.

## Evaluation

`evaluate.py --ckpt <path> --n_items 200` scores a checkpoint on
LibriSpeech-test and LJSpeech under: none, resample 8k, Gaussian noise (SNR 20),
median filter, low-pass 2k/4k, high-pass 500, reencode, compression, noise
suppression, phone call.

Runs at **batch size 1** deliberately. Batched evaluation pads to the batch max
and the encoder writes a watermark into that padding, which the decoder's mean
over the time axis reads back -- worth ~7-9 points that do not exist at
inference.

LJSpeech is at `/data/yizwen/LJSpeech-1.1_wav`: 22.05 kHz, 13100 clips split
across the root (7360) and `wavs/` (5740), no overlap. It needs its own loader
because `WavDataset` assumes the LibriSpeech split layout and does not resample
(`or_sample_rate == sample_rate` in the config).

### Previous run (b6hzmqft), for comparison

50 epochs, `lambda_m` effective 10, band 500-2000. Ended val SNR -8.86 dB with
the watermark ~9 dB louder than the speech and both discriminator losses at 0.

| distortion | LibriSpeech | LJSpeech |
| --- | --- | --- |
| none | 0.7035 | 0.7010 |
| resample_8k | 0.5322 | 0.5055 |
| gaussian_noise_20 | 0.5693 | 0.5487 |
| median_filter | 0.5317 | 0.5095 |
| low_pass_2k | 0.5176 | 0.4980 |
| low_pass_4k | 0.5312 | 0.5075 |
| high_pass_500 | **0.9729** | **0.9869** |
| reencode | 0.7035 | 0.7010 |
| compression | 0.7060 | 0.7085 |
| noise_suppression | 0.7055 | 0.7010 |
| phone_call | 0.6839 | 0.6879 |
| SNR dB | -6.31 | -4.10 |

LJSpeech tracked LibriSpeech almost exactly, so corpus shift is not a problem.
`high_pass_500` *beating* `none` is the diagnostic: watermark energy by band was
0-0.5k 0.40, 0.5-2k 0.15, 2-4k 0.13, **4-8k 0.31**, against speech energy of
0.45 / 0.52 / 0.03 / 0.006. Removing 0-500 Hz strips interference (+27 points);
removing 4-8 kHz strips signal (-17). Filters were verified with a white-noise
probe to rule out a filter bug.

## Open items

1. **Retune the loss weights.** Two full-scale runs now confirm it. The
   50-epoch no-VAD run ended at -8.86 dB val SNR, falling monotonically the
   whole way. `lambda_m: 10` vs `lambda_b: 1` / `lambda_a: 0.01` puts no
   effective bound on watermark amplitude. Use ~5-epoch runs (~3.5 h) to
   calibrate; accuracy is already >0.95 by epoch 5, so the only question is
   how much SNR can be bought back.

2. **Close the padding shortcut.** Measured worth ~7-9 accuracy points on
   matched audio, and it inflates every reported number. Fixed-length crops in
   `WavDataset.__getitem__` remove it at the source and also fix the loudness
   loss, which currently spends 84% of its softmax weight on padding segments.
   Until then, evaluate with batch size 1.

3. **Rotate the wandb API key.** It was committed in `config/train.yaml` and is
   still in git history (removing it from the working tree does not retract it).
   All those commits were on GitHub before this work started. Rotate at
   wandb.ai/settings.

4. **`future_amt_second` is 0.5, but this branch is named `from-0.25s`.**
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
