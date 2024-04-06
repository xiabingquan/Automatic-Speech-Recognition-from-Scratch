# coding=utf-8
# Contact: bingquanxia@qq.com

import os
import sys

import editdistance
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer, SubwordTokenizer
from dataloader import get_dataloader

from train import init_model

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python test.py <feature_extractor_type> <dataset_type> <checkpoint_path>")
        sys.exit(1)
    feature_extractor_type = sys.argv[1]
    dataset_type = sys.argv[2]
    ckpt_path = sys.argv[3]
    assert feature_extractor_type in ["linear", "resnet"]

    if dataset_type == "lrs2":
        t_ph = "./spm/lrs2/1000_bpe.model"
        audio_path_file = "./data/LRS2/test.paths"
        text_file = "./data/LRS2/test.text"
        lengths_file = "./data/LRS2/test.lengths"
    elif dataset_type == "librispeech":
        t_ph = "./spm/librispeech/1000_bpe.model"
        audio_path_file = "./data/LibriSpeech/test-clean.paths"
        text_file = "./data/LibriSpeech/test-clean.text"
        lengths_file = "./data/LibriSpeech/test-clean.lengths"

    # define tokenizer
    tokenizer = SubwordTokenizer(t_ph)

    # load data
    with open(audio_path_file) as f:
        audio_paths = f.read().splitlines()
    with open(text_file) as f:
        transcripts = f.read().splitlines()
    with open(lengths_file) as f:
        wav_lengths = f.read().splitlines()
    wav_lengths = [float(length) for length in wav_lengths]

    # define dataloader
    batch_size = 1
    batch_seconds = 100000      # unlimited
    data_loader = get_dataloader(
        audio_paths, transcripts, wav_lengths, tokenizer, batch_size, batch_seconds, shuffle=False
    )

    # define model
    vocab = tokenizer.vocab
    enc_dim = 256
    num_enc_layers = 12
    num_dec_layers = 6
    model = init_model(vocab, enc_dim, num_enc_layers, num_dec_layers, feature_extractor_type)
    model.eval()
    ckpt = torch.load(ckpt_path)
    missing, unexpected = model.load_state_dict(ckpt)
    print(f"Missing keys: {missing}. Unexpected: {unexpected}", flush=True)
    if torch.cuda.is_available():
        model.cuda()

    sos_id = tokenizer.sos_id
    eos_id = tokenizer.eos_id

    # main loop
    max_decode_len = 100        # the maximum length of the decoded sequence
    tot_err = 0
    tot_words = 0
    with torch.no_grad():
        print(f"index  |  ground truth  |  prediction  |  WER (Word Error Rate)", flush=True)
        for i, data in enumerate(data_loader):
            fbank_feat, feat_lens, _, _ = data
            assert fbank_feat.size(0) == 1
            gt = transcripts[i]
            ys_in_pad = torch.tensor([[sos_id]]).long()  # (1, 1)
            # greedy search
            while True:
                if torch.cuda.is_available():
                    fbank_feat = fbank_feat.cuda()
                    feat_lens = feat_lens.cuda()
                    ys_in_pad = ys_in_pad.cuda()
                logits = model(fbank_feat, feat_lens, ys_in_pad)
                logits = logits[0, -1]      # (vocab,)
                y_hat = logits.argmax(-1)
                if y_hat == eos_id:
                    break
                ys_in_pad = torch.cat([ys_in_pad, y_hat.view(1, -1)], dim=-1)
                if len(ys_in_pad[0]) > max_decode_len:
                    break
            ys_in_pad = ys_in_pad[:, 1:]  # [: 1:]: remove sos_id
            pred = tokenizer.detokenize(ys_in_pad[0].tolist())
            n_err = editdistance.eval(gt.split(), pred.split())
            n_wrd = len(gt.split())
            wer = n_err / n_wrd
            print(f"{i:05d}  |  {gt}  |  {pred}  |  {wer:.4f}", flush=True)
            tot_err += n_err
            tot_words += n_wrd
    wer = tot_err / tot_words
    print(f"WER: {wer:.4f}", flush=True)
