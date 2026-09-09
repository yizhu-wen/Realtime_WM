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

## Best checkpoint so far (2026-09-09)

`results/ckpt/pth/MSE_loudness_split_frequency_adaptive_soft_vad_phone_distortion_ep_60_2025-10-24_12_08_46.pth.tar`
-- the VAD-era model, epoch 60, added by the user. Loads **strictly** into the
current architecture (0 missing, 0 unexpected on both encoder and decoder). Its
encoder state_dict carries 30 `vad.*` tensors, since silero is an nn.Module
submodule.

It is far better than anything trained in this session: **0.98 clean accuracy
at +37 to +43 dB SNR**, i.e. accurate *and* inaudible. Compare the two 50-epoch
runs, which reached either 0.70 accuracy at -8.9 dB, or nothing at +131 dB.

| distortion | LibriSpeech-dev | LibriSpeech-test | LJSpeech |
| --- | --- | --- | --- |
| none | 0.9840 | 0.9804 | 0.9894 |
| resample_8k | 0.6975 | 0.6955 | 0.7503 |
| gaussian_noise_20 | 0.5615 | 0.5432 | 0.5291 |
| median_filter | 0.9235 | 0.9005 | 0.9141 |
| low_pass_2k | 0.6930 | 0.6930 | 0.7327 |
| low_pass_4k | 0.7260 | 0.7226 | 0.7704 |
| high_pass_500 | 0.9835 | 0.9809 | 0.9889 |
| reencode | 0.9835 | 0.9804 | 0.9889 |
| compression | 0.8995 | 0.9015 | 0.9141 |
| noise_suppression | 0.9360 | 0.9266 | 0.9839 |
| phone_call | 0.4825 | 0.4930 | 0.4854 |
| phone_call_legacy | 0.9255 | 0.9206 | 0.9156 |
| SNR dB | 38.41 | 37.08 | 42.97 |

### The two phone rows

`phone_call` uses the current `dl.py` channel: causal RIR (`mode="full"`) and a
300-3400 Hz band. `phone_call_legacy` (attack 33) is the channel this
checkpoint was trained against: centred RIR (`mode="same"`) and 500-2000 Hz.

The RIR mode is what decides it, not the band. On 80 test utterances:

| variant | acc |
| --- | --- |
| 300-3400, causal | 0.4582 |
| 500-2000, causal | 0.4848 |
| 500-2000, `mode="same"` | 0.9165 |
| 300-3400, `mode="same"` | 0.8899 |

`mode="same"` shifts the output by half the RIR length (~145 ms) relative to
`mode="full"`. Read 0.92 as "robust to its training channel" and 0.48 as "does
not transfer to a physically correct one".

### The decoder matches a channel signature, it does not extract robustly

A full factorial over the phone chain, 149 test utterances, 95% CI:

| condition | acc |
| --- | --- |
| none | 0.9812 +/-0.0069 |
| RIR+noise only, no band | 0.8698 +/-0.0171 |
| **RIR+noise + bp 500-2000 (exact training channel)** | **0.9195 +/-0.0138** |
| RIR+noise + bp 300-3400 | 0.8738 +/-0.0169 |
| bp 500-2000 only (no RIR) | 0.6450 +/-0.0243 |
| bp 300-3400 only (no RIR) | 0.7027 +/-0.0232 |

Two results that cannot both come from signal content:

1. **Adding a bandpass to RIR+noise raises accuracy**, 0.8698 -> 0.9195. A
   filter can only discard information, so it can only help by making the input
   look more like training data.
2. **The better band flips with the RIR.** With the RIR, 500-2000 beats
   300-3400 (0.9195 vs 0.8738). Without it, 300-3400 beats 500-2000 (0.7027 vs
   0.6450) -- the physically expected direction. Non-overlapping CIs both ways.

So the ~4.6 point drop when widening 500-2000 to 300-3400 is channel mismatch,
not lost watermark. The decoder is tuned to the exact chain it trained on
(centred RIR + noise + 500-2000). Removing the RIR alone costs 27 points
(0.9195 -> 0.6450).

Consistent with this: 92.5% of the watermark's energy sits below 300 Hz, which
every one of these bands discards. Energy is not information, but the decoder
is clearly not relying on the bulk of what the encoder emits.

**Practical consequence:** `phone_call_legacy` = 0.92 is an in-distribution
number and overstates real phone robustness. Training-channel diversity
(randomised RIR, band edges and codecs) is what would make it mean something.

### Restoring VAD needed three pieces, not one

The merge brought back the `forward` block, `self.vad` and the import, but not
the `__init__` assignments for `smooth_chunks`, `dilate_chunks`,
`target_smooth_ms`, `target_dilate_ms`, `tau`, nor those five config keys (both
removed in `e86ae20`). `Encoder(...)` constructed fine and only failed at
`forward` with `AttributeError: 'Encoder' object has no attribute
'smooth_chunks'`. All restored, plus `silero-vad` back in requirements.txt.

## BASELINE: the training recipe behind the best checkpoint

Starting point for further work. Checkpoint
`MSE_loudness_split_frequency_adaptive_soft_vad_phone_distortion_ep_60_2025-10-24_12_08_46.pth.tar`,
governing commit **`ccc9ab9`** (2025-10-18, the last before the checkpoint's
timestamp).

Provenance is confirmed, not guessed: the checkpoint's Adam state holds
**209640 steps**, and 209640 / 60 epochs = 3494 = ceil(27952 / 8), where 27952
is exactly the LibriSpeech train split after the >=3.0 s length filter. That
also pins `delay + future = 1.0 s`, since any other pair changes the filter and
hence the file count.

### Data
- LibriSpeech train, `/data/yizwen/LibriSpeech_wav/train`, 16 kHz mono
- 27952 clips after the `2 s prefill + 0.5 delay + 0.5 future = 3.0 s` filter
- `data_percentage: 1.0`, `data_divider: 1`
- random crop when longer than `max_len` 176000: `randint(5 s, max_len)`
- batch 8, zero-padded to the batch maximum by `collate_fn`
- **no data augmentation**; `select_aug_mode`, `n_max_aug` and `aug.*` are inert

### Framing
- `n_fft 322`, `hop 160`, `win 322` -> 162 bins
- `audio_prefilling 2.00 s` -> `voice_prefilling` 204 frames
- `delay_amt_second 0.5` -> 51 frames; `future_amt_second 0.5` -> 50 frames
- `offset_samples` 40480; message length 10 bits

### Architecture
- `conv2`, block `skip`, `layers_CE 3`, `layers_EM 4`, `hidden_dim 64`
- encoder reads 204 frames, writes 51, with a 500 ms gap
- **VAD gating ON** -- silero, threshold 0.50, `tau 0.15`,
  `target_smooth_ms 40`, `target_dilate_ms 15`, smooth/dilate counts derived;
  dynamic floor `[0.05, 0.20]` from frame RMS normalised by the utterance max
- **`mask = stft_result != 0`** present (suppresses the watermark over padding)
- discriminator ON (`adv: True`); **not saved** in this checkpoint

### Decoder-side channel (`distortion: true`)
Applied every step to the distorted head:
1. RIR: torchaudio demo RIR, **`mode="same"`** (acausal, ~145 ms pre-echo)
2. noise: `randn_like`, SNR `randint(20, 26)` dB
3. bandpass **500-2000 Hz**, hardcoded in `Decoder.__init__` -- *not* the
   `aug.cutoff_freq_*` config keys, which are inert

### Optimisation
- Adam `lr 1e-4`, `betas (0.9, 0.98)`, `eps 1e-9`, `weight_decay 0`
- `StepLR(step_size 5000, gamma 0.98)` stepped once per epoch -> no decay in 60
- grad clip `max_norm 1.0`, separately for encoder+decoder and discriminator
- 60 epochs, checkpoint every 5

### Loss
`Loss_identity` = waveform MSE + message MSE (both decoder heads) +
`TFLoudnessRatio(n_bands=16)`.

- `lambda_e 1.0`, `lambda_b 1.0`, `lambda_a 0.01`
- **`lambda_m` = 10 on step 1, then 0.01 for every step after.** The
  `lambda_a = lambda_m = ...` line was present in the train, val *and* test
  loops at `ccc9ab9`, so the effective message weight is 0.01.
- `pre_step = 0`, so the warmup branch never fires

### The open puzzle

Effective `lambda_m = 0.01` produced this checkpoint (**+38 dB SNR, 0.98
acc**) and also produced the September 2026 collapse (**+131 dB SNR, 0.50
acc**, watermark below the 16-bit quantisation floor). Same weighting, opposite
outcome.

The difference is the VAD. Plausible mechanism, **not yet tested**: the VAD mask
already suppresses the watermark in non-speech, so `TFLoudnessRatio` -- which is
unbounded below and otherwise keeps paying the model to go quieter -- has much
less left to push on, and the encoder can hold usable amplitude inside speech.
Without the VAD the loudness term ran to zero amplitude unopposed.

One epoch (~40 min) with VAD on vs off at `lambda_m` 0.01 would settle it, and
it decides whether the VAD is perceptual polish or load-bearing.

### Known defects present in this recipe

Carried by the run that produced the checkpoint; all diagnosed later in this
file:
- discriminator stepped on gradients contaminated by `g_loss_adv`
- acausal RIR (`mode="same"`)
- dynamic floor normalised by the utterance-global RMS max (non-causal)
- losses computed over batch padding; ~84% of the loudness softmax weight lands
  on padding segments
- the checkpoint omits the discriminator, so adversarial training cannot resume

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
