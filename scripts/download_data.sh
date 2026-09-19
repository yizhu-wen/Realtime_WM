#!/usr/bin/env bash
# Fetch the two real-speech corpora and convert them to the flat 16 kHz wav
# layout the trainer and evaluators expect.
#
#   bash scripts/download_data.sh [DATA_ROOT]     # default ./data
#
# Produces
#   $DATA_ROOT/LibriSpeech_wav/{train,val,test}/<utt>.wav   16 kHz mono
#   $DATA_ROOT/LJSpeech-1.1_wav/<utt>.wav                   16 kHz mono
#
# train = train-clean-100, val = dev-clean, test = test-clean.
set -euo pipefail
ROOT="${1:-$(pwd)/data}"
mkdir -p "$ROOT" && cd "$ROOT"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }

fetch () {  # url, filename
  [ -f "$2" ] || { echo "downloading $2"; curl -L --fail -o "$2" "$1"; }
}

# ---------------------------------------------------------------- LibriSpeech
for spec in "train-clean-100 train" "dev-clean val" "test-clean test"; do
  set -- $spec; part=$1; split=$2
  out="$ROOT/LibriSpeech_wav/$split"
  if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null | head -1)" ]; then
    echo "LibriSpeech/$split already present, skipping"; continue
  fi
  fetch "https://www.openslr.org/resources/12/${part}.tar.gz" "${part}.tar.gz"
  [ -d "LibriSpeech/$part" ] || tar xzf "${part}.tar.gz"
  mkdir -p "$out"
  echo "converting $part -> $split (flac to 16 kHz wav, flattened)"
  find "LibriSpeech/$part" -name '*.flac' -print0 \
    | xargs -0 -P "$(nproc)" -I{} sh -c \
        'f="{}"; ffmpeg -v error -y -i "$f" -ar 16000 -ac 1 "'"$out"'/$(basename "${f%.flac}").wav"'
  echo "  $split: $(ls "$out" | wc -l) files"
done

# ------------------------------------------------------------------- LJSpeech
out="$ROOT/LJSpeech-1.1_wav"
if [ -d "$out" ] && [ "$(ls -A "$out" 2>/dev/null | head -1)" ]; then
  echo "LJSpeech already present, skipping"
else
  fetch "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2" "LJSpeech-1.1.tar.bz2"
  [ -d LJSpeech-1.1 ] || tar xjf LJSpeech-1.1.tar.bz2
  mkdir -p "$out"
  # LJSpeech ships at 22.05 kHz; the models run at 16 kHz
  echo "converting LJSpeech -> 16 kHz wav"
  find LJSpeech-1.1/wavs -name '*.wav' -print0 \
    | xargs -0 -P "$(nproc)" -I{} sh -c \
        'f="{}"; ffmpeg -v error -y -i "$f" -ar 16000 -ac 1 "'"$out"'/$(basename "$f")"'
  echo "  LJSpeech: $(ls "$out" | wc -l) files"
fi

cat <<MSG

Done. Point config/train.yaml at the LibriSpeech root:

  path:
    raw_path: "$ROOT/LibriSpeech_wav"

MSG
