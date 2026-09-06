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

## Last run: completed 50 epochs (2026-09-06)

First full run on the no-VAD architecture (`e86ae20`). Ran 31.9 h, 50/50
epochs, zero errors.

- wandb: https://wandb.ai/yizhuwenus-university-of-hawaii-system/real-time-voice-watermark/runs/b6hzmqft
- run name: `novad_50ep_lm10-lb1_delay0.5_future0.5_causalRIR`
- checkpoints: `results/ckpt/pth/none-conv2_ep_{5..50}_*.pth.tar` (11 files)
- final: `none-conv2_ep_50_2026-09-06_10_53_55.pth.tar`

Reported test line: `acc [0.9733, 0.9503]  snr -9.51 dB`

### Do not trust the headline accuracy

The reported number is measured on padded batches, and the encoder learned to
use the padding. Measured on the epoch-50 checkpoint, held-out test set:

| condition | acc |
| --- | --- |
| batch 8, natural lengths, padded (as reported) | 0.976 / 0.956 |
| batch 1, natural length, no padding | 0.758 / 0.750 |

Isolating padding alone -- identical 8 s of audio, identical messages, batch of
8, differing *only* in trailing zeros:

| condition | acc |
| --- | --- |
| no padding | 0.9175 / 0.8863 |
| same audio + 3 s of zero padding | 0.9900 / 0.9775 |

So padding is worth ~7-9 points on matched audio; the rest of the gap to 0.758
is utterance duration (shorter clips give fewer watermark chunks). Watermark
power confirms the mechanism: 2.98e-02 inside speech vs **2.61e-02 in the zero
padding** -- the encoder puts almost as much energy into silence as into
speech, and the decoder's `mean(dim=2)` reads it back.

This is the shortcut that `mask = stft_result != 0` had been suppressing.

### SNR is the bigger problem

Val SNR fell monotonically for all 50 epochs and never recovered:

| epoch | 1 | 6 | 16 | 26 | 36 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| val SNR dB | -1.57 | -2.80 | -5.82 | -6.89 | -7.90 | **-8.86** |

The watermark ends ~9 dB *louder* than the speech. Both discriminator losses
collapsed to 0.0 by epoch 36 -- it separates watermarked from cover trivially.
Removing the VAD gate made this worse, not better (previous VAD run reached
-3.46 dB by epoch 5; this one -2.80 at epoch 6 and kept going).

`lambda_m: 10` against `lambda_b: 1` and `lambda_a: 0.01` does not constrain
loudness at all, and the loudness term is itself crippled by padding (84% of
its softmax weight lands on padding segments).

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
