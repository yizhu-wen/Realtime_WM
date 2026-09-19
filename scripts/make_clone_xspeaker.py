"""Cross-speaker voice cloning: each clip speaks one utterance's text in a
*different* speaker's voice. No real audio survives -- every sample is model
output. This is the voice-cloning threat model AudioSeal motivates.

Pairing is a cyclic shift over the sorted speaker list, which guarantees no
self-pairs and makes every speaker donate its voice exactly as often as it
supplies content. The donor utterance within a speaker is chosen by an
id-seeded RNG, so the pairing is reproducible and independent of pool order.

    python scripts/make_clone_xspeaker.py --librispeech data/LibriSpeech_wav \
        --transcripts data/LibriSpeech --out data/clone_xspeaker_wav --n 2000

Needs: pip install f5-tts       (~10 s per clip on a GPU; resumable)

Note: the published set used F5-TTS as an open stand-in for Voicebox, whose
weights were never released. Absolute numbers will not match papers that used
Voicebox; state the substitution if you publish.
"""
import argparse, glob, json, os, random, sys

import numpy as np
import soundfile as sf

SR = 16000


def load_transcripts(root):
    """LibriSpeech ships *.trans.txt per chapter: '<utt-id> THE TEXT'."""
    tr = {}
    for f in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        for line in open(f):
            uid, _, text = line.strip().partition(" ")
            tr[uid] = text.lower()
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True, help="root with val/ and test/ wavs")
    ap.add_argument("--transcripts", required=True,
                    help="untarred LibriSpeech root holding *.trans.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--min_sec", type=float, default=3.0)
    ap.add_argument("--max_sec", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = load_transcripts(a.transcripts)
    if not tr:
        sys.exit(f"no *.trans.txt under {a.transcripts}")

    files = sorted(sum((glob.glob(os.path.join(a.librispeech, s, "*.wav"))
                        for s in ("val", "test")), []))
    pool = [f for f in files
            if a.min_sec <= sf.info(f).frames / SR <= a.max_sec
            and os.path.splitext(os.path.basename(f))[0] in tr]
    random.Random(a.seed).shuffle(pool)
    pool = sorted(pool[: a.n])

    by_spk = {}
    for f in pool:
        by_spk.setdefault(os.path.basename(f).split("-")[0], []).append(f)
    speakers = sorted(by_spk)
    # cyclic shift: speaker i donates its voice to speaker i+1's text, so no
    # clip is self-paired and every speaker appears equally on both sides
    donor_of = {s: speakers[(i + 1) % len(speakers)] for i, s in enumerate(speakers)}
    print(f"{len(pool)} clips, {len(speakers)} speakers -> {a.out}")

    from f5_tts.api import F5TTS
    tts = F5TTS(model="F5TTS_v1_Base")

    os.makedirs(a.out, exist_ok=True)
    meta = os.path.join(a.out, "metadata.jsonl")
    done = set()
    if os.path.exists(meta):                       # resumable
        done = {json.loads(l)["id"] for l in open(meta)}
        print(f"  resuming, {len(done)} already generated")

    with open(meta, "a") as fh:
        for i, f in enumerate(pool):
            uid = os.path.splitext(os.path.basename(f))[0]
            if uid in done:
                continue
            spk = uid.split("-")[0]
            donor = donor_of[spk]
            # id-seeded choice keeps the pairing reproducible
            rng = random.Random(f"{uid}:{a.seed}")
            ref = rng.choice(by_spk[donor])
            ref_id = os.path.splitext(os.path.basename(ref))[0]
            try:
                wav, sr, _ = tts.infer(ref_file=ref, ref_text=tr[ref_id],
                                       gen_text=tr[uid], nfe_step=32, remove_silence=False)
                wav = np.asarray(wav, dtype=np.float32)
                if sr != SR:
                    import scipy.signal as ss
                    wav = ss.resample_poly(wav, SR, sr).astype(np.float32)
                sf.write(os.path.join(a.out, uid + ".wav"), wav, SR, subtype="PCM_16")
                fh.write(json.dumps(dict(
                    id=uid, content_id=uid, content_speaker=spk,
                    speaker=donor, prompt_speaker=donor, prompt_id=ref_id,
                    prompt_path=ref, prompt_transcript=tr[ref_id],
                    real_path=f, transcript=tr[uid], label="synthetic",
                    generator="f5-tts-xspeaker:F5TTS_v1_Base:nfe32",
                    mode="xspeaker", ratio=1.0, path=f"{uid}.wav",
                    sample_rate=SR, num_samples=len(wav),
                    duration=round(len(wav) / SR, 4),
                    source_duration=round(sf.info(f).frames / SR, 4))) + "\n")
                fh.flush()
            except Exception as e:
                print(f"  {uid} failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pool)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
