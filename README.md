# Automatic-Speech-Recognition-from-Scratch
An minimal Seq2Seq example of Automatic Speech Recognition (ASR) based on Transformer

Before launch training, you should download the train and test sub-sets of [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html),
and prepare `./data/LRS2/train.paths`、`./data/LRS2/train.text`、`./data/LRS2/train.lengths` with the format that  `train.py` requires.

Training: `python3 ./train.py`

Inference: `python3 ./test.py <ckpt_path>`


# Contact
bingquanxia AT qq.com
