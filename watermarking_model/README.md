# Timbre watermarking model
This is the complete code of the watermarking model part. Visit our [website](https://timbrewatermarking.github.io/samples.html) for audio samples.


# How to use
## Dependencies

You can setup the conda environment and install the Python dependencies with
```
git clone https://github.com/TimbreWatermarking/TimbreWatermarking.git
cd TimbreWatermarking/watermarking_model
conda create -n timbrewatermark python=3.8.13
source activate timbrewatermark
pip install -r requirements.txt
```
You may need to manually install the appropriate version of pytorch following the rules from [pytorch](https://pytorch.org/get-started/previous-versions).

## Dataset

You need to download the dataset used in the paper and extract it to an appropriate location.
You can use the script provided here to automatically download and process the [LibriSpeech dataset](https://www.openslr.org/12):
```
cd dataset 
sh ./sh.sh
```
For the [LJSpeech dataset](https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2), you can download it manually. 



## Inference

Modify 'raw_path' in `config/train.yaml` to point to your LibriSpeech path. The file structure is as follows, and the complete file structure can be found in `dataset/LibriSpeech_wav_structure.txt`.
```
LibriSpeech_wav
├── test
│   ├── 1089-134686-0000.wav
│   ├── 1089-134686-0001.wav
│   ├── 1089-134686-0002.wav
│   ├── 1089-134686-0003.wav
│   ├── 1089-134686-0004.wav
│   ├── 1089-134686-0005.wav
│   └── ......
├── train
│   ├── 103-1240-0000.wav
│   ├── 103-1240-0001.wav
│   ├── 103-1240-0002.wav
│   ├── 103-1240-0003.wav
│   ├── 103-1240-0004.wav
│   ├── 103-1240-0005.wav
│   └── ......
└── val
    ├── 1272-128104-0000.wav
    ├── 1272-128104-0001.wav
    ├── 1272-128104-0002.wav
    ├── 1272-128104-0003.wav
    ├── 1272-128104-0004.wav
    ├── 1272-128104-0005.wav
    └── ......
```

## Training

You can use the following command to retrain the watermarking model:
```
python train.py -p config/process.yaml -m config/model.yaml -t config/train.yaml
```
