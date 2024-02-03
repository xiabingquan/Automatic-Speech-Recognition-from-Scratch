# coding=utf-8
# Contact: bingquanxia@qq.com

import os
import sys

import editdistance
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer
from dataloader import get_dataloader

from train import init_model

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python test.py <checkpoint_path>")
        sys.exit(1)
    ckpt_path = sys.argv[1]

    tokenizer = CharTokenizer()

    # load data
    with open("./data/LRS2/test.paths") as f:
        audio_paths = f.read().splitlines()
    with open("./data/LRS2/test.text") as f:
        transcripts = f.read().splitlines()
    with open("./data/LRS2/test.lengths") as f:
        wav_lengths = f.read().splitlines()
    wav_lengths = [float(length) for length in wav_lengths]

    # define dataloader
    batch_size = 1
    batch_seconds = 100000
    data_loader = get_dataloader(
        audio_paths, transcripts, wav_lengths, tokenizer, batch_size, batch_seconds, shuffle=False
    )

    # define model
    model = init_model(tokenizer.vocab)
    model.eval()
    ckpt = torch.load(ckpt_path)
    missing, unexpected = model.load_state_dict(ckpt)
    print(f"Missing keys: {missing}. Unexpected: {unexpected}")
    if torch.cuda.is_available():
        model.cuda()

    sos_id = tokenizer.sos_id
    eos_id = tokenizer.eos_id

    # main loop
    max_decode_len = 100        # the maximum length of the decoded sequence
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            fbank_feat, feat_lens, _, ys_out_pad = data
            assert fbank_feat.size(0) == 1
            # gt = transcripts[i].lower()
            gt = tokenizer.detokenize(ys_out_pad[0, :-1].tolist())
            ys_in_pad = torch.tensor([[sos_id]]).long()  # (1, 1)
            # greedy search
            while True:
                if torch.cuda.is_available():
                    fbank_feat = fbank_feat.cuda()
                    feat_lens = feat_lens.cuda()
                    ys_in_pad = ys_in_pad.cuda()
                # print(ys_in_pad.size(), fbank_feat.size(), feat_lens.size())
                logits = model(fbank_feat, feat_lens, ys_in_pad)
                logits = logits[0, -1]      # (vocab,)
                y_hat = logits.argmax(-1)
                if y_hat == eos_id:
                    break
                ys_in_pad = torch.cat([ys_in_pad, y_hat.view(1, -1)], dim=-1)
                if len(ys_in_pad[0]) > max_decode_len:
                    break
            ys_in_pad = ys_in_pad[:, 1:]  # remove sos_id
            pred = tokenizer.detokenize(ys_in_pad[0].tolist())
            wer = editdistance.eval(gt.split(), pred.split()) / len(gt.split())
            print(f"{i:05d}  |  {gt}  |  {pred}  |  {wer:.4f}")
