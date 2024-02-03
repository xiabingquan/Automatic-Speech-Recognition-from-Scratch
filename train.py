# coding=utf-8
# Contact: bingquanxia@qq.com

import os

import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer
from dataloader import get_dataloader
from models import Encoder, Decoder, Transformer
from models import LinearFeatureExtractionModel


def init_model(vocab_size):
    fbank_dim = 80
    enc_dim = 256
    num_heads = enc_dim // 64
    num_layers = 6
    max_seq_len = 4096
    feature_extractor = LinearFeatureExtractionModel(fbank_dim, enc_dim)
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=num_layers, enc_dim=enc_dim, num_heads=num_heads, dff=2048, tgt_len=max_seq_len
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=num_layers, dec_dim=enc_dim, num_heads=num_heads, dff=2048, tgt_len=max_seq_len,
        tgt_vocab_size=vocab_size
    )
    model = Transformer(feature_extractor, encoder, decoder, enc_dim, vocab_size)
    return model


if __name__ == "__main__":

    # define tokenizer
    tokenizer = CharTokenizer()

    # load data
    with open("./data/LRS2/train.paths") as f:
        audio_paths = f.read().splitlines()
    with open("./data/LRS2/train.text") as f:
        transcripts = f.read().splitlines()
    with open("./data/LRS2/train.lengths") as f:
        wav_lengths = f.read().splitlines()
    wav_lengths = [float(length) for length in wav_lengths]

    # create checkpoint directory
    ckpt_dir = "./.checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    batch_size = 64
    batch_seconds = 256         # depends on your GPU memory
    data_loader = get_dataloader(
        audio_paths, transcripts, wav_lengths, tokenizer, batch_size, batch_seconds, shuffle=True
    )

    # define model
    model = init_model(tokenizer.vocab)
    print(model)
    model.train()
    # DataParallel for multi-gpu
    if torch.cuda.device_count() > 1:
        dp = True
        model = nn.DataParallel(model)
    else:
        dp = False
    if torch.cuda.is_available():
        model.cuda()

    # define optimizer and scheduler
    max_lr = 4e-4
    num_epoch = 20
    num_warmup = 10000
    pcb = num_warmup / (len(data_loader) * num_epoch)       # percentage of warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=len(data_loader), epochs=num_epoch,
        pct_start=pcb, anneal_strategy="cos",
    )

    # define loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)        # -1: ignore padding

    # main loop
    pbar = tqdm.tqdm(range(len(data_loader)), desc="Training")
    for epoch in range(1, num_epoch + 1):

        tot_loss = 0.
        data_loader.dataset.shuffle()

        for i, batch in enumerate(data_loader, start=1):
            # get batch data
            fbank_feat, feat_lens, ys_in_pad, ys_out_pad = batch
            if torch.cuda.is_available():
                fbank_feat = fbank_feat.cuda()
                feat_lens = feat_lens.cuda()
                ys_in_pad = ys_in_pad.cuda()
                ys_out_pad = ys_out_pad.cuda()

            # forward
            logits = model(fbank_feat, feat_lens, ys_in_pad)

            # calculate loss
            logits = logits.view(-1, logits.size(-1))
            ys_out_pad = ys_out_pad.view(-1).long()
            loss = criterion(logits, ys_out_pad)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # refresh progress bar
            tot_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{tot_loss / i:.2f}",
                "epoch": f"{epoch}/{num_epoch}",
            })
            pbar.update(1)
        pbar.reset()

        # save model
        torch.save(
            model.module.state_dict() if dp else model.state_dict(),
            os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
        )
