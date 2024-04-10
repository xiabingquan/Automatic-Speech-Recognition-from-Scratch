# Automatic-Speech-Recognition-from-Scratch

## Description
A minimal Seq2Seq example of Automatic Speech Recognition (ASR) based on Transformer

It aims to serve as a thorough tutorial for new beginners who is interested in training ASR models or other sequence-to-sequence models, complying with the blog in this link [包教包会！从零实现基于Transformer的语音识别(ASR)模型😘](https://zhuanlan.zhihu.com/p/648133707)

It contains almost everything you need to build a simple ASR model from scratch, such as training codes, inference codes, checkpoints, training logs and inference logs.

With this repository, you are expected to learn:
- How to build a Transformer model from scratch;
- How to apply Transformer into ASR task;
- How to pre-process and load audio data;
- How to create subword-based tokenizers and use them to process text data;
- How to train ASR models with [Model Parallel](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html);
- How to perform inference with greedy search and beam search.

To be as readable as possible, this repository does not contain complex components such as Distributed Data Parallel, language model restoring, CTC prefix beam search and so on. 

If you are looking for a high-level ASR library that supports multiple model architecture, decoding algorithms, and training frameworks, this repo may not be the best choice.
However, if you are eager to learn the basic stuff of ASR, this repo will NOT let you down.

Have fun! 🦦

## Data preprocessing

We use the audio part of [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) as our dataset.

Before launch training, you should download the train and test sub-sets of LRS2,
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

For convenience, we have prepared the three files above, the only thing you need to do is to place audio files consistent with `./data/LRS2/train.paths`. There you are ready to go.

> 💡 If you have difficulty in accessing dataset LRS2, you may use other ASR datasets, such as [LibriSpeech](https://www.openslr.org/12) or [TEDLIUM-v3](https://www.openslr.org/51/).
> However, in our preliminary experiments of LibriSpeech, we found that the model fails to converge under the default settings. You may need to modify the training or model hyper-parameters if necessary.

## Build tokenizers
Before training, you also need to prepare tokenizers.
In the [blog](https://zhuanlan.zhihu.com/p/648133707), we use char-based tokenizers.
However, considering that subword-based tokenizers are more often used in the ASR task, we use subword-based tokenizers instead.

Run `build_spm_tokenizer.sh` to build your subword-based tokenizer. You should replace the script's argument `save_prefix` and `txt_file_path` to fit your own data.

We have already provided tokenizers, located in the directory `spm/lrs2`. You could use them directly.


## training
Usage: `python train.py <feature_extractor_type> <dataset_type>`

We support two types of feature extractors: linear layer and 1D-ResNet18.
> The 1D-ResNet18 is based on the implementation of this [repo](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks).

We support two types of dataset: LRS2 and LibriSpeech.

For example, `python3 ./train.py resnet lrs2`.

The training logs are located in the `log` directory, containing the loss history and model details.

We highly encourage users to thoroughly read the codes if they want to customize their own datasets or understand the details of the training process.


## Inference
Usage: `python test.py <feature_extractor_type> <dataset_type> <checkpoint_path>`

For example, `python3 ./test.py resnet lrs2 ./ckpts/resnet_lrs2_epoch050.pt`

> Use the Linux command `cat` to merge checkpoint shards.
> For example, `cat resnet_lrs2_epoch050.pt.shard* > ./resnet_lrs2_epoch050.pt`

The checkpoints are located in the `ckpts` directory, containing both the linear and 1D-ResNet feature extractors.

The inference logs are located in the `log` directory, containing predictions of each sample.

We support two types of decoding algorithm: greedy search and beam search, both implemented inside `test.py`.

> The log files ends with `lrs2.test.log` contains the inference results of greedy search, while those named with the pattern `test.bms*.log` corresponds to beam search, the number `*` standing for the *beam size* argument used during inference.


## Warning
This repository is slightly different from the [blog](https://zhuanlan.zhihu.com/p/648133707) mentioned above in the following aspects.
- We use pre-norm instead of post-norm;
- We use subword-based tokenizers instead of char-based tokenizers;
- We add support of the feature extractor of 1D-ResNet.

## Contact
bingquanxia AT qq.com

