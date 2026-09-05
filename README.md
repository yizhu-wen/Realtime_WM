# Real-time speech watermarking

Streaming-capable neural speech watermarking. An encoder embeds an `n`-bit
message into speech using only a short lookahead, so it can run on a live
audio stream; a decoder recovers the message from the watermarked audio after
transmission-style distortion (room impulse response, additive noise, bandpass).

See [`watermarking_model/README.md`](watermarking_model/README.md) for setup,
dataset preparation, and training.

```bash
cd watermarking_model
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml
```

## Relation to prior work

This is a fork of [TimbreWatermarking](https://github.com/TimbreWatermarking/TimbreWatermarking)
(NDSS 2024), which this project adapts to the real-time / streaming setting.
The upstream voice-cloning experiment code has been removed; see that
repository if you need it.

```
@inproceedings{timbrewatermarking-ndss2024,
  title = {Detecting Voice Cloning Attacks via Timbre Watermarking},
  author = {Liu, Chang and Zhang, Jie and Zhang, Tianwei and Yang, Xi and Zhang, Weiming and Yu, Nenghai},
  booktitle = {Network and Distributed System Security Symposium},
  year = {2024},
  doi = {10.14722/ndss.2024.24200},
}
```
