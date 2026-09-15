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

### Two chance epochs are normal warmup -- judge at epoch 3, not epoch 1

The baseline run is `kcf7c7ol` on wandb (60 epochs, 63.3 h). Its own trajectory
settles the puzzle: it *looks* collapsed for two epochs and then breaks out.

| epoch | baseline `kcf7c7ol` | collapsed `oljqa8wp` |
| --- | --- | --- |
| 1 | 66.53 dB, acc 0.5015 | 70.55 dB, acc 0.4996 |
| 2 | 61.14 dB, acc 0.4953 | 84.77 dB, acc 0.4977 |
| 3 | **40.40 dB, acc 0.6767/0.9034** | 91.21 dB, acc 0.5027 |
| 10 | 39.26 dB, acc 0.8191/0.8848 | 104.14 dB, acc 0.4958 |
| 60/50 | 38.73 dB, acc 0.8815/0.9288 | 131.84 dB, acc 0.4989 |

**Falling val SNR over epochs 1-3 means it is breaking out; rising means it is
diverging.** An earlier note here claimed epoch 1 alone diagnosed the collapse.
That was wrong -- epoch 1 is indistinguishable between the two.

**The VAD is not the explanation either.** Tested 2026-09-09: with the VAD
restored at `lambda_m` 0.01, epoch 1 gave SNR 69.93 / acc 0.5014, matching the
VAD-less collapse. So the hypothesis that the VAD was load-bearing against
`TFLoudnessRatio` is disproved.

What actually differs between `kcf7c7ol` and `oljqa8wp` is therefore still
open, and the candidates are the remaining recipe deltas: the 500-2000 vs
300-3400 decoder band, the `stft_result != 0` mask, and the discriminator
gradient fix.

Also note the baseline's *own* numbers are more modest than the checkpoint
evaluation suggests: val acc 0.8815/0.9288 at epoch 60, against 0.98 measured
by `evaluate.py` on 200 dev utterances. Different message draws and a
200-utterance prefix, so not contradictory, but do not quote 0.98 as the
training-time figure.

### Known defects present in this recipe

Carried by the run that produced the checkpoint; all diagnosed later in this
file:
- discriminator stepped on gradients contaminated by `g_loss_adv`
- acausal RIR (`mode="same"`)
- dynamic floor normalised by the utterance-global RMS max (non-causal)
- losses computed over batch padding; ~84% of the loudness softmax weight lands
  on padding segments
- the checkpoint omits the discriminator, so adversarial training cannot resume

## Settled: the encoder VAD gate is what makes lambda_m 0.01 trainable

Five full-dataset runs, same effective `lambda_m` of 0.01 throughout. Val SNR
at epochs 1-3:

| run | VAD placement | band | RIR | ep1 | ep2 | ep3 | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kcf7c7ol` baseline | encoder, adaptive soft | 500-2000 | same | 66.53 | 61.14 v | 40.40 | **works** 38.7 dB, 0.88/0.93 |
| `oljqa8wp` | none | 500-2000 | causal | 70.55 | 84.77 ^ | 91.21 | collapsed, 131 dB |
| `hhqkp4ee` | encoder, adaptive soft | 300-3400 | same | 69.93 | 41.58 v | 40.35 | **works**, killed ep24 at 0.91/0.999 |
| `y76dzghh` | channel, after bandpass | 300-3400 | same | 70.67 | 85.95 ^ | 93.98 | collapsed |
| `g8sau4cz` | none | 300-3400 | same | 65.92 | 86.39 ^ | 96.27 | collapsed |

**At `lambda_m` 0.01: encoder VAD present -> converged 3/3 (`kcf7c7ol`,
`hhqkp4ee`, `rohklsax`). Absent or channel-side -> collapsed 5/5.** Band
(500-2000 vs 300-3400) and RIR mode (same vs causal) both vary *within* each
group, so neither explains the split.

**The `lambda_m` 0.01 scope is essential -- the VAD is not required to learn.**
Three runs with no VAD at all learned the message perfectly well at
`lambda_m` 10; they just produced an audible watermark:

| run | VAD | lambda_m | SNR dB | acc |
| --- | --- | --- | --- | --- |
| `wgr5miqz` | none | 10 | **-3.46** | 0.9289/0.9687 |
| `7fm0oqki` | none | 10 | **-4.23** | 0.9490/0.9835 |
| `b6hzmqft` | none | 10 | **-8.86** | 0.9774/0.9584 |

So the two knobs trade against each other, and the real statement is about
*which combinations give accuracy **and** inaudibility*:

| | `lambda_m` 0.01 | `lambda_m` 10 |
| --- | --- | --- |
| encoder VAD | **works: ~0.9 acc at +38 to +41 dB** | untested |
| no VAD | collapses to a zero watermark | learns, but -3 to -9 dB (audible) |
| channel VAD | collapses | untested |

Only 3 of 11 full runs are both accurate and inaudible, and all three have an
encoder-side gate at `lambda_m` 0.01. The top-right and bottom-right cells have
never been run.

Two later runs extend this (2026-09-11):

| run | VAD placement | ep2 | ep3 | outcome |
| --- | --- | --- | --- | --- |
| `pm7dkl4u` | none (rerun of `g8sau4cz`) | 85.47 | 97.48 | collapsed -- reproduces to ~1 dB |
| `ns4an8k8` | **adaptive soft**, channel after bandpass | 87.27 | 92.86 | collapsed |

`ns4an8k8` is the decisive one. The channel-side gate has now been tested with
**both** forms -- plain hard (`y76dzghh`) and adaptive soft (`ns4an8k8`) -- and
both collapse, while the same two gate forms in the *encoder* both converge
(`kcf7c7ol`/`hhqkp4ee` adaptive, `rohklsax` plain). So what matters is **where**
the gate sits, not what shape it has:

|  | encoder | channel |
| --- | --- | --- |
| adaptive soft | converged | **collapsed** |
| plain hard | converged | **collapsed** |

The gate has to constrain *what gets embedded*, not *what gets received*. A
channel-side mask leaves the encoder free to embed in silence, and the soft
mask's 0.05 floor -- which attenuates rather than removes, keeping gradients
flowing everywhere -- does not rescue it.

`ns4an8k8` ran to 20 epochs and its evaluation is the reference "fully
degenerate" row: **every distortion returns the identical accuracy** (dev
0.5075, test 0.5156, LJSpeech 0.5100) at SNR ~113 dB. Identical rather than
merely chance-level means the decoder output does not depend on its input at
all -- it emits a constant, and the figure is just chance agreement with the
random message. Useful as a sanity check: if a future evaluation shows the same
number in all 13 rows, the model is dead, not weak.

**The mechanism is NOT established.** An earlier version of this note asserted
that the encoder gate relieves pressure from `TFLoudnessRatio`. Three probes
were run on 2026-09-10 and none support it:

| probe | result |
| --- | --- |
| where the loudness loss puts its softmax weight | consistent: silence takes 97.9% without the gate (l_ratio +1.36 dB) vs 0% with it |
| per-term gradient norm on encoder params | **contradicts**: loudness dominates in both (96.6% vs 98.0%) and is *larger* with the gate, loud/msg 247x vs 359x |
| watermark RMS after 60 steps of the real loss | **contradicts**: shrinks 0.204x without the gate, 0.224x with it -- no difference |

Caveat: 60 steps is ~1.7% of an epoch and the trajectories separate around
epoch 2-3 (~7000-10000 steps), so these probes may simply not reach the regime
where it happens. They refute the stated mechanism at initialisation; they do
not rule out one that emerges later.

One lead, not an answer: with the gate the loudness loss goes negative and
stays there (-0.0034, stable); without it, it oscillates positive (+0.0129).
The gated model settles into a configuration that satisfies the loudness term
and the ungated one does not.

To resolve it properly, log per-term gradient norms and watermark RMS every N
steps through the first three epochs of both configs.

Diagnose at epoch 3, roughly 2 h. Epoch 1 is worthless -- `g8sau4cz` had the
most baseline-like epoch 1 of any run (65.92 vs the baseline's 66.53) and still
collapsed.

### Answered: a plain hard gate is enough (2026-09-10)

`rohklsax` (`plainVAD_20ep_lmeff0.01_band300-3400_sameRIR`) trained the
configuration that had been committed but never run: silero's raw `p > 0.5`
decision as a hard 0/1 mask, no sigmoid/tau, no smoothing, no dilation, no RMS
dynamic floor. Everything else matched `hhqkp4ee`.

| epoch | plain gate | baseline (adaptive soft) |
| --- | --- | --- |
| 1 | 58.02, acc 0.502 | 66.53, acc 0.502 |
| 2 | 60.70, acc 0.496 | 61.14, acc 0.495 |
| 3 | 56.89, acc 0.500 | **40.40, acc 0.677/0.903** |
| 4 | **40.22, acc 0.690/0.874** | 40.37, acc 0.747/0.912 |

It breaks out one epoch later and lands on the same operating point. So the
**presence** of an encoder gate is what matters; the adaptive machinery is
perceptual refinement, not what makes training viable at `lambda_m` 0.01.

**Warning about the epoch-3 test.** A hardcoded `SNR < 55` threshold labelled
this run "DIVERGING" at epoch 3 (56.89 dB), which was wrong -- SNR had *fallen*
from 60.70 and was nowhere near the 91-96 dB of the real collapses. Acting on
that label would have killed a working run. Use the direction of SNR plus
whether `msg_loss` has moved off 2.0, not a single threshold, and give a slow
run until epoch 5 before judging.

### Previously untested gap (now closed by the above)

The **plain hard 0.5 VAD in the encoder** (commit `43b3d96`: encoder gate,
RIR -> noise -> bandpass, 300-3400) was committed and smoke-tested but never
trained. Every trained run used either the *adaptive soft* gate in the encoder
or no encoder gate at all. `git revert 0a0439a cb9e688` restores it.

### The other lever

`lambda_m` cannot currently be set: `train.py:332`
`lambda_a = lambda_m = train_config["optimize"]["lambda_a"]` pins it to 0.01
every step, so `config/train.yaml`'s `lambda_m: 10.` is inert. Removing that
line and setting `lambda_m` explicitly is required to test anything in the
0.01-10 bracket, where 0.01 collapses without a gate and 10 gave -8.9 dB.

## BEST MODEL: plainVAD ep20 (`rohklsax`, 2026-09-11)

`results/ckpt/pth/none-conv2_ep_20_*.pth.tar` from run `rohklsax`. Plain hard
0.5 encoder gate, telephony chain RIR(same) -> noise(20-26 dB) ->
bandpass(300-3400), effective `lambda_m` 0.01, 20 epochs.

Best of the session on almost every axis. LibriSpeech-test, batch size 1:

| distortion | plainVAD ep20 | adaptive ep20 | baseline ep60 |
| --- | --- | --- | --- |
| none | **0.9784** | 0.9307 | 0.9809 |
| resample_8k | **0.9302** | 0.9332 | 0.6849 |
| gaussian_noise_20 | **0.5940** | 0.5769 | 0.5648 |
| median_filter | **0.9472** | 0.8593 | 0.8925 |
| low_pass_2k | 0.7432 | 0.7603 | 0.6869 |
| low_pass_4k | **0.9377** | 0.9221 | 0.7327 |
| high_pass_500 | 0.9683 | 0.9256 | 0.9789 |
| reencode | 0.9784 | 0.9307 | 0.9809 |
| compression | **0.9709** | 0.8648 | 0.8925 |
| noise_suppression | **0.9417** | 0.8839 | 0.9302 |
| phone_call (causal) | 0.5643 | 0.5714 | 0.4784 |
| phone_call_legacy | 0.6749 | 0.7352 | 0.9291 |
| phone_call_trainmatch | **0.9302** | 0.9030 | 0.8889 |
| SNR dB | 38.30 | 41.13 | 37.07 |

LJSpeech tracks it: 0.9910 clean at 40.80 dB, and LibriSpeech-dev 0.9835 at
38.97 dB.

Against the **adaptive gate at the same 20 epochs and same band** -- the clean
single-variable comparison -- the plain gate is +5.8 points on average across
non-band-limiting channels (clean +4.8, compression +10.6, median +8.8,
noise-suppression +5.8) and level on band-limiting (-0.2). It gives up 2.8 dB
of SNR.

Against the **ep-60 baseline**, it is +16.9 points on band-limiting channels
(resample_8k +24.5, low_pass_4k +20.5) at +1.2 dB better SNR, in a third of the
epochs. `phone_call_legacy` -25.4 is not a regression: that is the baseline's
own 500-2000 training channel. On each model's own channel,
`phone_call_trainmatch` 0.9302 vs the baseline's `phone_call_legacy` 0.9291.

Still weak: Gaussian noise (~0.59) and the causal `phone_call` (~0.56). The
first is inherent to a watermark ~38 dB below the speech; the second is the
acausal-RIR mismatch, since training uses `mode="same"`.

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

## Detection metrics and baseline comparison (2026-09-14)

Three scripts, run from `watermarking_model/`:

| script | what | output |
| --- | --- | --- |
| `detect_eval.py` | per-utterance detection scores, one method x one corpus | `results/detect/<Method>_<Corpus>.npz` |
| `imperceptibility.py` | per-utterance SNR and wideband PESQ | `results/imp/<Method>_<Corpus>.npz` |
| `detect_report.py` | aggregates both into Excel + LaTeX | `results/evals/` |

Five methods: RT-SW (this repo, `rohklsax`) plus AudioSeal, WavMark, Timbre and
SilentCipher, wrapped by `/data/yizwen/wm_shift_exp/methods.py`. Four corpora:
LibriSpeech dev 2303, LJSpeech 2700, clone_xspeaker 1624, resynth_hifigan.

### Protocol

Per clip, per distortion: the **positive** is `decode(distort(embed(x)))` scored
against the embedded message; the **negative** is `decode(distort(x))` on the
same clip, unwatermarked, scored against an independent random message. The
statistic is the bit-match count, which every method supports whatever its
payload. Under H0 it is Binomial(n_bits, 0.5), so the negatives form an
empirical null.

Pairing positives and negatives on the same clip controls for speaker and
content, but makes the classes correlated -- the AUC bootstrap therefore
resamples **clips**, not the two classes independently.

The set is exactly balanced by construction, which does not actually matter:
TPR and FPR are each computed within one class, so prevalence cancels. It would
matter for classification accuracy, precision or F1, none of which are reported.

### The payload-size trap

The statistic is an integer, so attainable FPRs form a ladder whose spacing is
set by payload size. Near 1%:

| payload | rungs |
| --- | --- |
| 10 bits (RT-SW, Timbre) | 1.07% at t=9, **0.098%** at t=10 -- nothing between |
| 16 bits (AudioSeal, WavMark) | 1.06% at t=13, 0.21% at t=14 |
| 40 bits (SilentCipher) | lands close to 1% |

A deterministic threshold therefore holds a 10-bit method to a rate up to 10x
stricter than a 40-bit one, and their TPRs are **not comparable**. Worse, it is
unstable: as the pooled null grew from 7900 to 11890 negatives RT-SW's
threshold flipped 9 -> 10 and its clean TPR moved 0.935 -> 0.874, on sampling
noise alone.

`detect_report.py` reports both operating points. Use `--tex_tpr exact` (a
randomised Neyman-Pearson rule: reject above t, and at t-1 with probability
gamma) for anything comparing methods. AUC is threshold-free and immune.

### Things that will bite

- **Same clips for every method.** The duration filter is RT-SW's 3 s minimum
  for all five. Filtering per method handed the baselines 400 extra short
  LibriSpeech utterances and made the table meaningless.
- **Timbre collides with this repo.** TimbreWatermarking is a fork and ships a
  `distortions` package with the same module names but an STFT taking
  `[B, 1, T]` where ours takes `[B, T]`. Whichever imports first wins for both.
  `detect_eval.py` builds our chain first, then drops `distortions` from
  `sys.modules` and this repo from `sys.path`.
- **SilentCipher's payload is 5 bytes, not 40 bits.** Null messages are drawn
  per method by `random_msg()`; drawing 40 bits makes `bits_of_bytes` return
  320 and the comparison fails to broadcast.
- **Clip order is shuffled** (fixed seed 0, separate from `--seed`) before
  truncation, because the sorted prefix is one block of speakers. Combined with
  the 200-clip checkpointing, an interrupted run is a valid smaller-N sample.
- **The GPU is the bottleneck, not the CPU.** 12 concurrent jobs serialized
  almost completely (RT-SW 1.1 -> 10 s/clip) for ~20% aggregate gain. Use 3.
- **WavMark costs ~15 s/clip**, 20x RT-SW, because its decode is a 0.1 s-shift
  sliding sync search. `decode_batch_size` does not help (10 -> 400 was
  *slower*, identical accuracy).
- **Two environments.** RT-SW needs `timbrewm` conda; the baselines need
  `/data/yizwen/.venv-wm`. `pesq` was installed into both on 2026-09-14.
- **RT-SW needs `conv2_mel_modules.py` at `1add787`** to load `rohklsax`
  strictly. Check it out before running, restore with
  `git restore --source=HEAD --staged --worktree` after.

### Imperceptibility

SNR is `10*log10(mean(x^2)/mean((y-x)^2))`, the same definition `evaluate.py`
uses. PESQ is ITU-T P.862 wideband, defined only at 16 kHz, so SilentCipher's
44.1 kHz signals are resampled for PESQ alone; SNR stays native.

First figures (LibriSpeech-dev): SilentCipher 48.9 dB / 4.55, RT-SW 38.8 /
4.12, WavMark 35.9 / 3.95, Timbre 27.0 / 3.59, AudioSeal 26.4 / **4.31**.
AudioSeal is why SNR alone is not enough -- it has the worst SNR and the second
best PESQ, because it shapes the watermark perceptually rather than minimising
its energy.

RT-SW's measured 38.82 dB (dev) and 40.89 dB (LJSpeech) match the 38.97 and
40.80 recorded for this checkpoint above, which validates the measurement.

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
