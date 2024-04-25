# coding=utf-8
# Contact: bingquanxia@qq.com

import os
import sys
from typing import Union, Optional

import editdistance
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import CharTokenizer, SubwordTokenizer
from dataloader import get_dataloader

from train import init_model


@torch.no_grad()
def greedy_search(model, fbank_feat, feat_lens, sos_id, eos_id, max_decode_len):
    assert fbank_feat.size(0) == 1
    ys_in_pad = torch.tensor([[sos_id]]).long()     # (1, 1)
    if torch.cuda.is_available():
        ys_in_pad = ys_in_pad.cuda()

    while True:
        logits = model(fbank_feat, feat_lens, ys_in_pad)
        logits = logits[0, -1]  # (bs, vocab)
        y_hat = logits.argmax(-1)
        if y_hat == eos_id:
            break
        ys_in_pad = torch.cat([ys_in_pad, y_hat.view(1, -1)], dim=-1)
        if len(ys_in_pad[0]) > max_decode_len:
            break
    pred_tokens = ys_in_pad[0, 1:]  # [: 1:]: remove sos_id

    return [pred_tokens]


@torch.no_grad()
def beam_search_serial(
    model, fbank_feat, feat_lens, sos_id, eos_id, max_decode_len,
    bms=10,
):

    class Hypothesis(object):
        def __init__(self, eos_id: int, tokens: list, score: float = 0.):
            self.eos_id = eos_id
            self.tokens = tokens
            self.score = score

        @classmethod
        def build_from_prev(cls, prev_hyp, token: int, logp: float):
            return Hypothesis(
                prev_hyp.eos_id, prev_hyp.tokens + [token], prev_hyp.score + logp
            )

        def finished(self):
            return self.tokens[-1] == self.eos_id

    # init hypotheses
    hyps = [Hypothesis(eos_id, [sos_id], 0.) for _ in range(bms)]
    for i in range(1, bms):
        hyps[i].score = float("-inf")

    # get encoder output
    assert fbank_feat.size(0) == 1, "Only support batch size 1."
    enc_out, feat_lens = model.get_encoder_output(fbank_feat, feat_lens)

    # main loop
    for i in range(max_decode_len):

        # check whether all beams are finished
        if all([h.finished() for h in hyps]):
            break

        bs = 1      # batch size
        # iterate over all beams
        new_hyps = []
        for h in hyps:

            # forward
            l = torch.tensor(h.tokens, device=enc_out.device).view(1, -1)
            dec_mask = model.get_subsequent_mask(bs, l.size(1), l.device)
            dec_enc_mask = model.get_enc_dec_mask(bs, enc_out.size(1), feat_lens, l.size(1), l.device)
            logits = model.get_logits(enc_out, l, dec_mask, dec_enc_mask)
            logits = logits[:, -1]          # (1, T, vocab) -> (1, vocab)
            logp = F.log_softmax(logits, dim=-1)

            # local pruning: prune non-topk scores
            topk_logp, topk_idxs = logp.topk(k=bms, dim=-1)  # (1, vocab) -> (1, bms)
            topk_logp, topk_idxs = topk_logp.view(-1), topk_idxs.view(-1)   # (bms,), (bms,)

            # masked finished beams
            if h.finished():
                topk_logp[0] = 0.
                topk_logp[1:] = float("-inf")
                topk_idxs.fill_(eos_id)

            # calculate scores of new beams
            for j in range(bms):
                new_hyps.append(
                    Hypothesis.build_from_prev(h, topk_idxs[j].item(), topk_logp[j].item())
                )

        # global pruning
        new_hyps = sorted(new_hyps, key=lambda x: x.score, reverse=True)
        hyps = new_hyps[:bms]

    # get the best hyp
    best_hyp = max(hyps, key=lambda x: x.score)
    pred_tokens = best_hyp.tokens[1:]  # [: 1:]: remove sos_id
    pred_tokens = [t for t in pred_tokens if t != eos_id]  # remove eos_id

    return [pred_tokens]


@torch.no_grad()
def beam_search_parallel(
        model, fbank_feat, feat_lens, sos_id, eos_id, max_decode_len,
        bms=10,
):

    def mask_finished_scores(scores, end_flag, inf=-float("inf")):
        """
        Example of end_flag:
            0
            1
            0
            1
            1
        Corresponding mask `mask_to_inf`:
            0 0 0 0 0
            0 1 1 1 1
            0 0 0 0 0
            0 1 1 1 1
            0 1 1 1 1
        Corresponding mask `mask_to_zero`:
            0 0 0 0 0
            1 0 0 0 0
            0 0 0 0 0
            1 0 0 0 0
            1 0 0 0 0
        In the above case, there're five samples and five beams.
        The second and the fivth samples have mask_to_zero beam searching.

        """
        rns, bms = scores.size()
        assert end_flag.size(0) == rns and end_flag.ndim == 1
        zero_mask = scores.new_zeros(rns, 1)
        mask_to_zero = torch.cat([end_flag.view(rns, 1), zero_mask.repeat(1, bms - 1)], dim=-1)  # (rns, bms)
        mask_to_inf = torch.cat([zero_mask, end_flag.view(rns, 1).repeat(1, bms - 1)], dim=-1)  # (rns, bms)
        scores = scores.masked_fill(mask_to_zero.bool(), 0.)
        scores = scores.masked_fill(mask_to_inf.bool(), inf)
        return scores

    def mask_finished_preds(preds, end_flag, eos_id):
        # Force preds to be all `sos` for finished beams.
        rns, bms = preds.size()
        finished = end_flag.view(-1, 1).repeat(1, bms)  # (rns, bms)
        preds.masked_fill_(finished.bool(), eos_id)
        return preds

    # bms: beam size, rns: running size
    bs = fbank_feat.size(0)                         # batch size
    assert bs == 1, "For simplicity, we only support batch size 1."
    rns = bs * bms

    # init hypotheses, scores and flags
    hyps = torch.tensor([[sos_id]]).long().repeat(bs, 1)  # (bs, 1)
    hyps = hyps.unsqueeze(1).repeat(1, bms, 1).view(rns, 1)  # (rns, 1), the hypothesis of current beam
    scores = torch.zeros(bms).float()
    scores[1:] = float("-inf")
    scores = scores.repeat(bs, 1).view(rns)                     # (rns,), the scores of current beam
    end_flag = torch.zeros(rns).bool()                         # (rns,), whether current beam is finished
    if torch.cuda.is_available():
        hyps = hyps.cuda()
        scores = scores.cuda()
        end_flag = end_flag.cuda()

    # get encoder output
    enc_out, feat_lens = model.get_encoder_output(fbank_feat, feat_lens)
    enc_out = enc_out.unsqueeze(1).repeat(1, bms, 1, 1).view(rns, enc_out.size(-2), enc_out.size(-1))
    feat_lens = feat_lens.unsqueeze(1).repeat(1, bms).view(rns,)

    # main loop
    for i in range(max_decode_len):

        # check whether all beams are finished
        if end_flag.all():
            break

        # forward
        dec_mask = model.get_subsequent_mask(rns, hyps.size(1), hyps.device)
        dec_enc_mask = model.get_enc_dec_mask(rns, enc_out.size(1), feat_lens, hyps.size(1), hyps.device)
        logits = model.get_logits(enc_out, hyps, dec_mask, dec_enc_mask)  # (rns, T, vocab)
        logits = logits[:, -1]
        logp = F.log_softmax(logits, dim=-1)  # (rns, vocab)

        # local pruning: prune non-topk scores
        topk_logp, topk_idxs = logp.topk(k=bms, dim=-1)  # (rns, vocab) -> (rns, bms)

        # masked finished beams
        topk_logp = mask_finished_scores(topk_logp, end_flag)
        topk_idxs = mask_finished_preds(topk_idxs, end_flag, eos_id)

        # calculate scores of new beams
        scores = scores.view(rns, 1)
        scores = scores + topk_logp  # (rns, 1) + (rns, bms) -> (rns, bms)
        scores = scores.view(bs, bms * bms)

        # global pruning
        scores, offset_k_idxs = scores.topk(k=bms, dim=-1)  # (bs, bms)
        scores = scores.view(rns, 1)
        offset_k_idxs = offset_k_idxs.view(-1)

        # calculate the predicted token at current decoding step
        base_k_idxs = torch.arange(bs, device=scores.device) * bms * bms
        # wrong implementation:
        # base_k_idxs = base_k_idxs.repeat(bms).view(-1)
        # correct implementation:
        base_k_idxs = base_k_idxs.unsqueeze(-1).repeat(1, bms).view(-1)
        # e.g. base_k_idxs: (0, 0, 0, 9, 9, 9, 81, 81, 81)
        best_k_idxs = base_k_idxs + offset_k_idxs.view(-1)
        best_k_pred = torch.index_select(topk_idxs.view(-1), dim=-1, index=best_k_idxs)

        # retrive the old hypotheses of best k beams
        best_hyp_idxs = best_k_idxs.div(bms, rounding_mode="floor")
        last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)  # (rns, i)

        # concat the old hypotheses with the new predicted token
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (rns, i)

        # refresh end_flag
        end_flag = torch.eq(hyps[:, -1], eos_id).view(-1)

    # get the best hyp
    scores = scores.view(-1, bms)  # (rns, bms)
    _, best_hyp_idxs = scores.topk(k=1, dim=-1)  # (bs, 1)
    best_hyp_idxs = best_hyp_idxs.view(-1)
    idxs = torch.arange(bs, device=scores.device) * bms
    idxs = idxs.unsqueeze(1).repeat(1, 1).view(-1)
    best_hyp_idxs += idxs
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)

    pred_tokens = best_hyps[:, 1:]      # [: 1:]: remove sos_id
    pred_tokens = [hyp[hyp != eos_id].tolist() for hyp in pred_tokens]            # remove eos_id

    return pred_tokens


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python test.py <feature_extractor_type> <dataset_type> <checkpoint_path>")
        sys.exit(1)

    print(f"ARGS: {sys.argv}")

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
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

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
    batch_seconds = 100000  # unlimited
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
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt)
    print(f"Missing keys: {missing}. Unexpected: {unexpected}", flush=True)
    if torch.cuda.is_available():
        model.cuda()

    # decoding-related constants
    # search_strategy = "greedy"
    search_strategy = "beam_search"
    beam_size = 10
    sos_id = tokenizer.sos_id
    eos_id = tokenizer.eos_id

    # main loop
    max_decode_len = 100  # the maximum length of the decoded sequence
    tot_err = 0
    tot_words = 0
    print(f"index  |  ground truth  |  prediction  |  WER (Word Error Rate)", flush=True)

    for i, (fbank_feat, feat_lens, ys_in, ys_out) in enumerate(tqdm.tqdm(data_loader)):
        assert fbank_feat.size(0) == 1, "Only support batch size 1."

        if torch.cuda.is_available():
            fbank_feat = fbank_feat.cuda()
            feat_lens = feat_lens.cuda()

        if search_strategy == "greedy":
            pred_tokens = greedy_search(model, fbank_feat, feat_lens, sos_id, eos_id, max_decode_len)
        elif search_strategy == "beam_search":
            beam_search = beam_search_parallel
            # beam_search = beam_search_serial
            pred_tokens = beam_search(
                model, fbank_feat, feat_lens, sos_id, eos_id, max_decode_len,
                bms=beam_size
            )
        else:
            raise ValueError(f"Invalid search strategy: {search_strategy}")
        pred_tokens = pred_tokens[0]

        pred = tokenizer.detokenize(pred_tokens)
        gt = transcripts[i]

        n_err = editdistance.eval(gt.split(), pred.split())
        n_wrd = len(gt.split())
        tot_err += n_err
        tot_words += n_wrd

        wer = n_err / n_wrd
        print(f"{i:05d}  |  {gt}  |  {pred}  |  {wer:.4f}", flush=True)

    wer = tot_err / tot_words
    print(f"WER: {wer:.4f}", flush=True)
