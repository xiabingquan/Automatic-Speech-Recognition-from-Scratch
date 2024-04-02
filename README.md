# Automatic-Speech-Recognition-from-Scratch
A minimal Seq2Seq example of Automatic Speech Recognition (ASR) based on Transformer

Before launch training, you should download the train and test sub-sets of [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html),
and prepare `./data/LRS2/train.paths`、`./data/LRS2/train.text`、`./data/LRS2/train.lengths` with the format that  `train.py` requires.

Each line in train.paths represents the local path of an audio file. 

Each line in train.text represents a text sentence. 

Each line in train.lengths represents an integer value indicating the length of the audio (number of seconds).

The following table suggests a minimal example of the above three files.
| train.paths | train.text       | train.lengths |
| ----------- | ---------------- | ------------- |
| 1.wav       | good morning     | 1.6         |
| 2.wav       | good afternoon   | 2         |
| 3.wav       | nice to meet you | 3.1         |
> 💡 If you have difficulty in accessing dataset LRS2, you may use other ASR datasets, such as [LibriSpeech](https://www.openslr.org/12) or [TEDLIUM-v3](https://www.openslr.org/51/)
> 
> Use `torchaudio`, `ffmpeg` or any other tools to get the length information of audio
>
> If you are experiencing convergence issue, try subword-based tokenizers ([ref](https://github.com/google/sentencepiece)) or more sophisticated feature extractors (e.g. [1D ResNet](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/0cc99fab046bd959578f2baac52e654746da4825/lipreading/models/resnet1D.py#L75)).

Training: `python3 ./train.py`

Inference: `python3 ./test.py <ckpt_path>`


# Contact
bingquanxia AT qq.com
