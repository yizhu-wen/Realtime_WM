"""Per-utterance detection scores for one watermarking method on one corpus.

Protocol, per clip and per distortion:
  positive : y = embed(x, msg);  decode(distort(y)) -> bits matching msg
  negative : no embedding;       decode(distort(x)) -> bits matching a fresh msg

The detection statistic is the **bit-match count**, which every method supports
regardless of payload size (RT-SW/Timbre 10 bits, AudioSeal/WavMark 16,
SilentCipher 40). Under H0 it is Binomial(n, 0.5), so the negatives give an
empirical null from which a threshold at a target FPR is calibrated.

Writes an .npz of raw per-utterance scores; aggregate with detect_report.py.

  python detect_eval.py --method RTSW --dataset LibriSpeech-dev --out x.npz
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# name -> (attack index in distortions.dl, ratio). "phone_call" is the
# centred-RIR 300-3400 channel (attack 34), the one the model trained against.
DISTORTIONS = [
    ("none", 0, 0),
    ("resample_8k", 8, 0),
    ("gaussian_noise_20", 9, 20),
    ("median_filter", 13, 3),
    ("low_pass_4k", 32, 0),
    ("high_pass_500", 15, 0),
    ("reencode", 27, 0),
    ("compression", 29, 0),
    ("noise_suppression", 28, 0),
    ("phone_call", 34, 0),
]

REPO = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "LibriSpeech-dev": "/data/yizwen/LibriSpeech_wav/val",
    "LJSpeech": "/data/yizwen/LJSpeech-1.1_wav",
    "clone_xspeaker": "/data/yizwen/clone_xspeaker_wav",
    "resynth_hifigan": "/data/yizwen/resynth_hifigan_wav",
}


# ---------------------------------------------------------------- methods


class RTSWMethod:
    """This repo's model. Needs the code version the checkpoint was trained
    with; the caller is responsible for checking that out first."""

    name, sr, n_bits = "RT-SW", 16000, 10

    def __init__(self, ckpt):
        import torch
        import yaml

        sys.path.insert(0, REPO)
        from model.conv2_mel_modules import Decoder, Encoder

        self.torch = torch
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pc = yaml.safe_load(open(os.path.join(REPO, "config/process.yaml")))
        mc = yaml.safe_load(open(os.path.join(REPO, "config/model.yaml")))
        tc = yaml.safe_load(open(os.path.join(REPO, "config/train.yaml")))
        n = tc["watermark"]["length"]
        self.enc = Encoder(pc, mc, tc, n).to(self.dev).eval()
        self.dec = Decoder(pc, mc, tc, n).to(self.dev).eval()
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        self.enc.load_state_dict(ck["encoder"])
        self.dec.load_state_dict(ck["decoder"])
        # the decoder's own training distortion must be off: the only channel
        # applied is the one under test
        self.dec.distortion = False
        assert n == self.n_bits, n

    def random_msg(self, rng):
        return rng.integers(0, 2, self.n_bits)

    def embed(self, x, rng):
        t = self.torch.from_numpy(x)[None].to(self.dev)
        msg = self.random_msg(rng)
        m = self.torch.from_numpy(msg).float().to(self.dev)[None, None] * 2 - 1
        with self.torch.no_grad():
            out = self.enc(t, m, 1)
        if out is None:
            return None, msg
        return (t + out[0])[0].cpu().numpy().astype(np.float32), msg

    def decode(self, x, msg):
        t = self.torch.from_numpy(np.ascontiguousarray(x))[None].to(self.dev)
        with self.torch.no_grad():
            d, _ = self.dec(t, 1)
        bits = (d.squeeze().cpu().numpy() >= 0).astype(int)
        return int((bits == msg).sum())


class BaselineMethod:
    """Wraps /data/yizwen/wm_shift_exp/methods.py, which returns bitacc."""

    def __init__(self, key):
        sys.path.insert(0, "/data/yizwen/wm_shift_exp")
        import methods

        self.key = key
        self.impl = methods.ALL[key]()
        self.name = self.impl.name
        self.sr = self.impl.sr
        self.n_bits = self.impl.n_bits

    def random_msg(self, rng):
        # SilentCipher's payload is 5 bytes that its wrapper expands to 40 bits;
        # every other method's payload is already bits.
        if self.key == "SilentCipher":
            return rng.integers(0, 256, 5)
        return rng.integers(0, 2, self.n_bits)

    def embed(self, x, rng):
        # Timbre's encoder returns [1, 1, T] where the others return [T];
        # flatten so the distortion chain always sees a bare waveform.
        y, msg = self.impl.embed(x, rng)
        return np.asarray(y, dtype=np.float32).reshape(-1), msg

    def decode(self, x, msg):
        r = self.impl.decode(x, msg)
        return int(round(r["bitacc"] * self.n_bits))


def method_sample_rate(method):
    """The method's working rate, without instantiating it (which would import
    its whole stack)."""
    if method == "RTSW":
        return RTSWMethod.sr
    sys.path.insert(0, "/data/yizwen/wm_shift_exp")
    import methods

    return methods.ALL[method].sr


# ---------------------------------------------------------------- data


def list_clips(dataset, sr, min_samples):
    # soundfile, not torchaudio.info: torchaudio >=2.9 removed info/load in
    # favour of torchcodec, and .venv-wm ships 2.11.
    import soundfile as sf

    root = DATASETS[dataset]
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not files:
        raise SystemExit(f"no wavs under {root}")
    keep, bad = [], 0
    for f in files:
        try:
            i = sf.info(f)
        except Exception:
            bad += 1
            continue
        # length after resampling to the method's rate must clear the minimum
        if i.frames / i.samplerate * sr >= min_samples:
            keep.append(f)
    if bad:
        print(f"  warning: {bad} files unreadable", flush=True)
    if not keep:
        raise SystemExit(
            f"{root}: {len(files)} wavs but none >= {min_samples/sr:.2f}s at {sr} Hz"
        )
    return keep


def load(path, target_sr):
    import soundfile as sf
    import scipy.signal as ss

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != target_sr:
        x = ss.resample_poly(x, target_sr, sr).astype(np.float32)
    return np.ascontiguousarray(x)


def save(out, m, dataset, names, pos, neg, used, done, scanned=0):
    """Atomic write, so a reader never sees a half-written checkpoint."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez(tmp, method=m.name, dataset=dataset, n_bits=m.n_bits, sr=m.sr,
             used=used, complete=bool(done), scanned=int(scanned),
             names=np.array(names),
             **{f"pos_{n}": np.array(pos[n]) for n in names},
             **{f"neg_{n}": np.array(neg[n]) for n in names})
    os.replace(tmp, out)
    if done:
        print(f"wrote {out}  ({used} clips)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["RTSW", "Timbre", "AudioSeal", "WavMark", "SilentCipher"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--n_items", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from distortions.dl import distortion
    import yaml

    # The distortion chain must run at the method's own sample rate, or its
    # filter cutoffs and resamplers are wrong. Read the rate off the class
    # rather than an instance, so the chain can be built before the method
    # exists (see the Timbre note below).
    sr = method_sample_rate(args.method)
    pc = yaml.safe_load(open(os.path.join(REPO, "config/process.yaml")))
    pc["audio"]["or_sample_rate"] = sr
    dist = distortion(pc)

    # TimbreWatermarking is a fork of this repo and ships its own `distortions`
    # package with the same module names but an incompatible STFT (its
    # transform takes [B, 1, T], ours takes [B, T]). Whichever loads first wins
    # for both, so build our chain, then drop the package from sys.modules and
    # take this repo off sys.path before Timbre imports its own copy.
    if args.method == "Timbre":
        for k in [k for k in sys.modules if k == "distortions" or
                  k.startswith("distortions.")]:
            del sys.modules[k]
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or os.getcwd()) != REPO]

    m = RTSWMethod(args.ckpt) if args.method == "RTSW" else BaselineMethod(args.method)
    assert m.sr == sr, (m.sr, sr)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Every method is scored on the same clips, so the minimum duration is
    # RT-SW's for all of them: it needs 2 s of prefill plus the delay and the
    # lookahead, where the baselines would accept anything over a second.
    # Filtering per method would hand the baselines 400 extra short LibriSpeech
    # utterances and make the comparison table meaningless.
    tc = yaml.safe_load(open(os.path.join(REPO, "config/train.yaml")))
    min_s = 2 + tc["watermark"]["delay_amt_second"] + tc["watermark"]["future_amt_second"]
    min_samples = int(min_s * m.sr)

    clips = list_clips(args.dataset, m.sr, min_samples)
    # Shuffle before truncating. The file lists are sorted, so a sequential
    # prefix is one contiguous block of speakers (LibriSpeech) or one stretch
    # of a single recording session (LJSpeech). The seed is fixed and separate
    # from --seed, so every method walks the same corpus in the same order and
    # partial runs stay comparable across methods.
    np.random.default_rng(0).shuffle(clips)
    if args.n_items:
        clips = clips[: args.n_items]
    print(f"{m.name} / {args.dataset}: {len(clips)} clips, sr {m.sr}, {m.n_bits} bits",
          flush=True)

    names = [d[0] for d in DISTORTIONS]
    pos = {n: [] for n in names}
    neg = {n: [] for n in names}
    rng = np.random.default_rng(args.seed)
    used = 0
    start = 0

    # Resume an interrupted run. The clip order is deterministic (shuffled with
    # a fixed seed), so the first `scanned` entries are exactly the ones already
    # attempted, and the message draws are replayed to keep the RNG aligned.
    # Without this a crash at clip 800 threw away every clip before it.
    if os.path.exists(args.out):
        try:
            prev = np.load(args.out, allow_pickle=True)
            have = "scanned" in prev.files
            if not bool(prev["complete"]) and have and int(prev["scanned"]) > 0:
                start = int(prev["scanned"])
                used = int(prev["used"])
                for nm in names:
                    pos[nm] = list(prev[f"pos_{nm}"])
                    neg[nm] = list(prev[f"neg_{nm}"])
                for _ in range(used):        # replay: one msg + one null per kept clip
                    m.random_msg(rng)
                    m.random_msg(rng)
                print(f"  resuming at clip {start} ({used} already scored)",
                      flush=True)
        except Exception as e:
            print(f"  could not resume ({type(e).__name__}), starting over",
                  flush=True)
            start, used = 0, 0

    def apply(sig, idx, ratio):
        t = torch.from_numpy(np.ascontiguousarray(sig))[None, None].to(dev)
        y = dist(t, attack_choice=idx, ratio=ratio)
        if y.dim() == 3:
            y = y.squeeze(1)
        y = y[0].detach().cpu().numpy().astype(np.float32)
        if y.shape[-1] != sig.shape[-1]:  # codecs/resamplers can shift length
            k = min(y.shape[-1], sig.shape[-1])
            y = y[:k]
        return np.ascontiguousarray(y)

    for i, f in enumerate(clips):
        if i < start:
            continue
        try:
            x = load(f, m.sr)
            y, msg = m.embed(x, rng)
            if y is None:
                continue
            neg_msg = m.random_msg(rng)  # independent null message, native format
            # Score the whole clip before committing any of it. A codec or
            # resampler that throws partway through would otherwise leave the
            # arrays ragged and, worse, silently break the positive/negative
            # pairing that the clip-level bootstrap relies on.
            p_i, n_i = {}, {}
            for nm, idx, ratio in DISTORTIONS:
                p_i[nm] = m.decode(apply(y, idx, ratio), msg)
                n_i[nm] = m.decode(apply(x, idx, ratio), neg_msg)
            for nm in p_i:
                pos[nm].append(p_i[nm])
                neg[nm].append(n_i[nm])
            used += 1
        except Exception as e:
            print(f"  clip {i} failed: {type(e).__name__}: {str(e)[:90]}", flush=True)
        # Checkpoint often. WavMark costs ~10 s of compute per second of
        # audio, so 200 clips is nearly two hours of silence with no partial
        # result and no way to measure the rate.
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(clips)} ({used} used)", flush=True)
            # Checkpoint. Because the clip order is shuffled, whatever has been
            # scored so far is a uniform random sample of the corpus, so a run
            # stopped early is a valid smaller-N result rather than a biased
            # prefix. Lets a slow method be cut short without losing its work.
            save(args.out, m, args.dataset, names, pos, neg, used,
                 done=False, scanned=i + 1)
            # Release cached blocks back to the driver. Even with
            # expandable_segments the caching allocator grows with job age --
            # one Timbre job went 3.5 GB to 9.5 GB over 80 minutes -- and with
            # twelve workers on one card that ends in cuFFT allocation
            # failures. Once per checkpoint is far too rare to cost throughput.
            torch.cuda.empty_cache()

    save(args.out, m, args.dataset, names, pos, neg, used,
         done=True, scanned=len(clips))


if __name__ == "__main__":
    main()
