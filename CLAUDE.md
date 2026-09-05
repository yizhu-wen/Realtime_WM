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

## Active run (started 2026-09-05 03:04)

Fresh 50-epoch run on the full dataset, with the VAD/mask/RIR changes in
`e86ae20`. Launched detached (`setsid`) so it survives this session.

- wandb: https://wandb.ai/yizhuwenus-university-of-hawaii-system/real-time-voice-watermark/runs/b6hzmqft
- run name: `novad_50ep_lm10-lb1_delay0.5_future0.5_causalRIR`
- log: `watermarking_model/results/log/novad_full_20260905_030352.log` (gitignored)
- main trainer pid at launch: 3857065
- settings: 28539 train files, batch 8, `lambda_m: 10`, `lambda_b: 1`,
  delay/future 0.5s, `adv: True`, `distortion: true`
- ETA ~38 h

Finding the pid: the trainer is the process orphaned by `setsid` (ppid 1);
`pgrep -f "python train.py"` also matches its 22 DataLoader workers, so
grabbing the first match gets a worker and looks like the run died.

```bash
for p in $(pgrep -f "python train.py -p config"); do
  [ "$(awk '{print $4}' /proc/$p/stat)" = 1 ] && echo $p; done
```

**Killing it needs the process group.** `persistent_workers=True` with
`num_workers=20` means killing the parent leaves ~40 orphaned workers holding
~53 GB of VRAM. Kill the group, then sweep by command line, then confirm the
GPU actually dropped to ~2 MiB.

### Previous run (superseded)

50-epoch run at `wgr5miqz`, killed by the user after 5 epochs (3h31m). It
showed message decoding works well at full scale (val acc 0.929/0.969 by epoch
5) while val SNR fell monotonically +0.59 -> -3.46 dB, and both discriminator
losses collapsed to ~1e-3. Its epoch-5 checkpoint is at
`results/ckpt/pth/none-conv2_ep_5_2026-09-05_01_38_25.pth.tar`, but it predates
`e86ae20` and was trained with VAD gating that no longer exists, so it is not a
valid warm start for the current architecture.

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
