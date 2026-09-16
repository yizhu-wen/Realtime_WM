"""Distortions used by the AudioSeal, Timbre and SilentCipher papers.

Kept separate from `dl.py` (user-supplied) so the existing evaluation path is
untouched. Every attack takes and returns a 1-D float32 numpy array at 16 kHz.

Parameter choices are stated in PARAMS below. Where a paper gives a figure it
is used; where it only names the attack, a conventional value is chosen and
noted, because the exact strength changes the numbers and should be visible
rather than buried.
"""

import os
import subprocess
import tempfile

import numpy as np

PARAMS = {
    # --- AudioSeal -------------------------------------------------------
    "boost":            "amplitude x1.2",
    "duck":             "amplitude x0.8",
    "mp3_64":           "MP3 64 kbps",
    "mp3_128":          "MP3 128 kbps",
    "mp3_256":          "MP3 256 kbps",
    "aac_64":           "AAC 64 kbps",
    "aac_128":          "AAC 128 kbps",
    "aac_256":          "AAC 256 kbps",
    "encodec":          "EnCodec 24 kHz, 6 kbps",
    "smoothing":        "moving average, window 40 samples (2.5 ms)",
    "speed_up":         "1.25x resample (pitch shifts too, as in the paper)",
    "pink_noise":       "pink noise at SNR 20 dB",
    # --- Timbre ----------------------------------------------------------
    "median_5":         "median filter, kernel 5 samples",
    "median_15":        "median filter, kernel 15 samples",
    "median_25":        "median filter, kernel 25 samples",
    "median_35":        "median filter, kernel 35 samples",
    "crop_middle":      "remove the central 10% of samples",
    "crop_end":         "remove the final 10% of samples",
    # --- SilentCipher ----------------------------------------------------
    "ogg":              "Ogg Vorbis, quality 4 (~128 kbps)",
    "time_jitter":      "drop/duplicate 1 sample every 1000 (0.1% jitter)",
    "quant_16bit":      "16-bit uniform quantisation",
    "speech_mix_-15dB": "another utterance mixed in at -15 dB",
    "sample_suppress":  "zero 0.1% of samples, uniformly at random",
}
ORDER = list(PARAMS)


# ------------------------------------------------------------------ helpers


def _ffmpeg_roundtrip(x, codec_args, ext, sr):
    """Encode through ffmpeg and read back, length-matched to the input."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.wav")
        enc = os.path.join(d, "enc." + ext)
        dec = os.path.join(d, "out.wav")
        sf.write(src, x, sr)
        for cmd in (["ffmpeg", "-y", "-v", "error", "-i", src, *codec_args, enc],
                    ["ffmpeg", "-y", "-v", "error", "-i", enc,
                     "-ar", str(sr), "-ac", "1", dec]):
            subprocess.run(cmd, check=True, capture_output=True)
        y, _ = sf.read(dec, dtype="float32")
    if y.ndim > 1:
        y = y[:, 0]
    # codecs pad with encoder delay; trim or zero-fill to the original length
    if len(y) >= len(x):
        return np.ascontiguousarray(y[: len(x)])
    return np.ascontiguousarray(np.pad(y, (0, len(x) - len(y))))


def _pink(n, rng):
    """Pink noise via 1/f spectral shaping of white noise."""
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.arange(len(W))
    f[0] = 1
    p = np.fft.irfft(W / np.sqrt(f), n=n)
    return p / (np.std(p) + 1e-12)


# ------------------------------------------------------------------ attacks


class PaperAttacks:
    def __init__(self, device="cuda:0", mixer_pool=None, seed=0, sr=16000):
        # SilentCipher runs at 44.1 kHz; every other method at 16 kHz. The
        # codec round-trips and EnCodec's resampling both need the true rate,
        # so it is a parameter rather than a module constant.
        self.sr = sr
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._encodec = None
        # utterances used as the interfering talker for speech mixing
        self.mixer_pool = mixer_pool or []

    # --- AudioSeal
    def boost(self, x):            return np.clip(x * 1.2, -1.0, 1.0).astype(np.float32)
    def duck(self, x):             return (x * 0.8).astype(np.float32)
    def mp3_64(self, x):           return _ffmpeg_roundtrip(x, ["-b:a", "64k"], "mp3", self.sr)
    def mp3_128(self, x):          return _ffmpeg_roundtrip(x, ["-b:a", "128k"], "mp3", self.sr)
    def mp3_256(self, x):          return _ffmpeg_roundtrip(x, ["-b:a", "256k"], "mp3", self.sr)
    def aac_64(self, x):           return _ffmpeg_roundtrip(x, ["-c:a", "aac", "-b:a", "64k"], "m4a", self.sr)
    def aac_128(self, x):          return _ffmpeg_roundtrip(x, ["-c:a", "aac", "-b:a", "128k"], "m4a", self.sr)
    def aac_256(self, x):          return _ffmpeg_roundtrip(x, ["-c:a", "aac", "-b:a", "256k"], "m4a", self.sr)

    def encodec(self, x):
        import torch, torchaudio
        if self._encodec is None:
            from encodec import EncodecModel
            m = EncodecModel.encodec_model_24khz()
            m.set_target_bandwidth(6.0)
            self._encodec = m.to(self.device).eval()
        t = torch.from_numpy(x)[None, None].to(self.device)
        t24 = torchaudio.functional.resample(t, self.sr, 24000)
        with torch.no_grad():
            enc = self._encodec.encode(t24)
            dec = self._encodec.decode(enc)
        y = torchaudio.functional.resample(dec, 24000, self.sr)[0, 0].cpu().numpy()
        k = min(len(y), len(x))
        out = np.zeros_like(x); out[:k] = y[:k]
        return out.astype(np.float32)

    def smoothing(self, x, w=None):
        # 2.5 ms, so the perceptual strength is the same at any sample rate
        w = w or max(3, int(round(0.0025 * self.sr)))
        k = np.ones(w, dtype=np.float32) / w
        return np.convolve(x, k, mode="same").astype(np.float32)

    def speed_up(self, x, rate=1.25):
        import scipy.signal as ss
        n = max(1, int(round(len(x) / rate)))
        return np.ascontiguousarray(ss.resample(x, n).astype(np.float32))

    def pink_noise(self, x, snr_db=20.0):
        n = _pink(len(x), self.rng)
        ps = float(np.mean(x ** 2))
        pn = ps / (10 ** (snr_db / 10.0))
        return (x + n * np.sqrt(pn)).astype(np.float32)

    # --- Timbre
    def _median(self, x, k):
        import scipy.signal as ss
        return ss.medfilt(x.astype(np.float64), kernel_size=k).astype(np.float32)

    def median_5(self, x):         return self._median(x, 5)
    def median_15(self, x):        return self._median(x, 15)
    def median_25(self, x):        return self._median(x, 25)
    def median_35(self, x):        return self._median(x, 35)

    def crop_middle(self, x, ratio=0.10):
        cut = int(len(x) * ratio)
        b = (len(x) - cut) // 2
        return np.ascontiguousarray(np.concatenate([x[:b], x[b + cut:]]))

    def crop_end(self, x, ratio=0.10):
        return np.ascontiguousarray(x[: len(x) - int(len(x) * ratio)])

    # --- SilentCipher
    def ogg(self, x):              return _ffmpeg_roundtrip(x, ["-c:a", "libvorbis", "-q:a", "4"], "ogg", self.sr)

    def time_jitter(self, x, period=None):
        """Drop or duplicate one sample every `period`, alternating, so the
        length stays put while local timing is perturbed."""
        period = period or max(2, int(round(self.sr / 16)))   # 0.1% of samples
        out = x.copy()
        idx = np.arange(period, len(x) - 1, period)
        for j, i in enumerate(idx):
            if j % 2 == 0:
                out[i:-1] = out[i + 1:]        # drop one sample
            else:
                out[i + 1:] = out[i:-1]        # duplicate one sample
        return np.ascontiguousarray(out.astype(np.float32))

    def quant_16bit(self, x):
        q = np.round(np.clip(x, -1.0, 1.0) * 32767.0) / 32767.0
        return q.astype(np.float32)

    def speech_mix(self, x, snr_db=-15.0):
        """Mix in another utterance at the given SNR. -15 dB means the
        interferer is 15 dB *below* the signal."""
        if not self.mixer_pool:
            return x.astype(np.float32)
        m = self.mixer_pool[self.rng.integers(len(self.mixer_pool))]
        if len(m) < len(x):
            m = np.tile(m, int(np.ceil(len(x) / len(m))))
        m = m[: len(x)]
        ps, pm = float(np.mean(x ** 2)), float(np.mean(m ** 2)) + 1e-12
        g = np.sqrt(ps / pm * 10 ** (snr_db / 10.0))
        return (x + m * g).astype(np.float32)

    def sample_suppress(self, x, frac=0.001):
        out = x.copy()
        k = max(1, int(len(x) * frac))
        out[self.rng.choice(len(x), k, replace=False)] = 0.0
        return out.astype(np.float32)

    # --- dispatch
    def apply(self, name, x):
        return {
            "boost": self.boost, "duck": self.duck,
            "mp3_64": self.mp3_64, "mp3_128": self.mp3_128, "mp3_256": self.mp3_256,
            "aac_64": self.aac_64, "aac_128": self.aac_128, "aac_256": self.aac_256,
            "encodec": self.encodec, "smoothing": self.smoothing,
            "speed_up": self.speed_up, "pink_noise": self.pink_noise,
            "median_5": self.median_5, "median_15": self.median_15,
            "median_25": self.median_25, "median_35": self.median_35,
            "crop_middle": self.crop_middle, "crop_end": self.crop_end,
            "ogg": self.ogg, "time_jitter": self.time_jitter,
            "quant_16bit": self.quant_16bit,
            "speech_mix_-15dB": self.speech_mix,
            "sample_suppress": self.sample_suppress,
        }[name](x)
