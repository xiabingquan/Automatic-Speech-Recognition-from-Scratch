import os
import sys
import logging
import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def rm_dub_and_blank(hyp: List[int], blank: int):
    """
    Remove dubplicated tokens and blanks. Used in CTC decoding.
    Notice that all hyps should be padded to the same length after this function called, since not all preds have the
    same length

    References: https://github.com/wenet-e2e/wenet/wenet/utils/common.py

    Args:
        hyp:
        blank: the index of blank sign

    Returns:

    """
    cur = 0
    max_len = len(hyp)
    new_hyp = []

    # A simpler implementation from fairseq.
    # References:
    # import itertools as it
    # def get_tokens(blank, idxs):
    #     """Normalize tokens by handling CTC blank, ASG replabels, etc."""
    #     idxs = (g[0] for g in it.groupby(idxs))
    #     idxs = filter(lambda x: x != blank, idxs)
    #     return torch.LongTensor(list(idxs))

    while cur < max_len:
        if hyp[cur] != blank:
            new_hyp.append(hyp[cur])
        prev_char = hyp[cur]
        while cur < max_len and hyp[cur] == prev_char:
            cur += 1
    if len(new_hyp) > 0 and new_hyp[-1] == blank:    # remove extra blank
        new_hyp == new_hyp[:-1]
    return new_hyp


class BeamSearch(nn.Module):
    def __init__(
            self, model: nn.Module, tokenizer, max_decode_len: int, bms: int,
            sos: int, eos: int, blank: int,
            ctc_beta=None, lm_beta=None, lm=None, lb_beta=None,
    ):
        """

        Args:
            model: model to be used for decoding
            tokenizer: the tokenizer of the model
            max_decode_len: the max length of decoded sequence
            bms: the beam size
            sos: the id of start of sentence
            eos: the id of end of sentence
            blank: the id of blank sign
            ctc_beta: the ratio of ctc
            lm_beta: the ratio of language model
            lm: language model
            lb_beta: the ratio of length bonus.
                When lb_beta is None or lb_beta == 0., no length bonus is used.
                When lb_beta > 0., we intend to shorten the decoded sequence.
                When lb_beta < 0., we intend to lengthen the decoded sequence.
        """
        super().__init__()
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

        self.max_decode_len = max_decode_len
        self.bms = bms          # beam_size
        self.sos, self.eos, self.blank = sos, eos, blank
        self.ctc_beta, self.lm_beta = ctc_beta, lm_beta
        self.lb_beta = lb_beta
        self.lm = lm
        logger.info(
            f"BeamSearch: beam_size={self.bms}, sos={sos}, eos={eos}, blank={blank}, "
            f"ctc_beta={ctc_beta}, lm_beta={lm_beta}, lm={lm}, lb_beta={lb_beta}."
        )
    
    @staticmethod
    def mask_finished_scores(scores, end_flag, device, inf=-float("inf")):
        """

        Only reserve one branch in those mask_to_zero beams.

        References: https://github.com/wenet-e2e/wenet/wenet/utils/common.py

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

        Args:
            scores: (rns, bms)
            end_flag: (rns, 1)
            device:
            inf:

        Returns:

        """
        bms = scores.shape[-1]
        zero_mask = torch.zeros_like(end_flag, device=device)  # (rns， 1)
        if bms > 1:
            mask_to_inf = torch.cat((zero_mask, end_flag.repeat((1, bms - 1))), dim=1).to(device)  # (rns, bms)
            mask_to_zero = torch.cat((end_flag, zero_mask.repeat((1, bms - 1))), dim=1).to(device)
        else:
            mask_to_inf = zero_mask
            mask_to_zero = end_flag
        scores.masked_fill_(mask_to_inf, inf)
        scores.masked_fill_(mask_to_zero, 0.)
        return scores

    @staticmethod
    def mask_finished_preds(preds, end_flag, eos, device):
        """

        Force preds to be all `sos` for finished beams.

        References: https://github.com/wenet-e2e/wenet/wenet/utils/common.py

        Args:
            preds: (rns, bms)
            end_flag: (rns, 1)
            eos:
            device:

        Returns:

        """
        bms = preds.shape[-1]
        finished = end_flag.repeat((1, bms))
        return preds.masked_fill_(finished, eos).to(device)

    def detokenize(self, preds):
        """

        Args:
            preds: (bs, T)

        Returns:

        """
        pred_lens = [(t != self.eos).sum(axis=-1) for t in preds]
        pred_sents = [self.tokenizer.detokenize(p[:p_len].tolist()) for p, p_len in zip(preds, pred_lens)]
        return pred_sents

    def forward(self, feats, feat_lens):
        """

        Args:
            feats:
            feat_lens:

        Returns:

        """
        pred_tokens = self.decode(feats, feat_lens)      # (bs, T)
        # return pred_tokens                            # for ONNX export
        pred_sents = self.detokenize(pred_tokens)
        return pred_sents

    @torch.inference_mode()
    def decode(self, feats, feat_lens):
        """

        Args:
            feats:
            feat_lens:

        Returns:

        """

        # constants
        bs = feats.size(0)
        device = feats.device
        bms = self.bms
        rns = bms * bs      # rns: running_size

        # init cache
        cache = self.model.init_cache(feats, feat_lens, self.bms)
        if self.ctc_beta is not None and 0. < self.ctc_beta < 1.:           # use ctc
            ctc_scores = self.model.init_ctc_cache(feats, feat_lens)        # B, T, vocab
        else:
            ctc_scores = None

        # init hyps and scores
        hyps = torch.ones((rns, 1), device=device).fill_(self.sos).long()
        scores = torch.zeros(bms, device=device)
        scores[0], scores[1:] = 0., -float("inf")
        scores = scores.repeat(bs).unsqueeze(1)     # (rns, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)

        # main loop
        max_len = min(feat_lens.max().item(), self.max_decode_len)
        for i in range(max_len):
            # # debug
            # pred_lens = [(t != self.eos).sum(axis=-1) for t in hyps]
            # pred_sents = [self.tokenizer.detokenize(p[:p_len].tolist()) for p, p_len in zip(hyps, pred_lens)]
            # print(f"{i:05d}:")
            # print('\n'.join(pred_sents) + '\n')
            # print()

            # stop if all beams in all batchs produce sos
            if end_flag.sum() == rns:
                break
            # get the logp of current step (we only need the last token of hyps)
            # logp, cache = self.model.forward_one_step(cache, hyps[:, [-1]])
            # Notes: why we do not directly use the above line?
            #  Because the above line will cause the error: the position embedding is not correct.
            logp, cache = self.model.forward_one_step(cache, hyps)
            # local pruning
            topk_logp, topk_idxs = logp.topk(k=bms, dim=-1)  # (rns, vocab) -> (rns, bms)
            topk_logp = self.mask_finished_scores(topk_logp, end_flag, device)
            topk_idxs = self.mask_finished_preds(topk_idxs, end_flag, self.eos, device)
            scores = scores + topk_logp             # broadcast add: (rns, 1) + (rns, bms) -> (rns, bms)
            scores = scores.view((bs, bms * bms))   # (rns, bms) -> (bs, bms * bms)
            # global pruning
            scores, offset_k_idxs = scores.topk(k=bms, dim=-1)  # scores after pruning: (bs, bms)
            scores = scores.view((-1, 1))  # scores: (rns, 1)
            # e.g. base_k_idxs: (0, 0, 0, 9, 9, 9, 27, 27, 27, 36, 36, 36)
            base_k_idxs = (torch.arange(bs, device=device) * bms * bms).unsqueeze(1).repeat((1, bms)).view(-1)
            best_k_idxs = base_k_idxs + offset_k_idxs.view(-1)
            best_k_pred = torch.index_select(topk_idxs.view(-1), dim=-1, index=best_k_idxs)  # word idxs of current step
            best_hyp_idxs = best_k_idxs.div(bms, rounding_mode="floor")
            last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)  # (rns, i)

            # concat history hyps and current word
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1)  # (rns, step)

            # refresh end_flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)
        # tell the model that the inference is ended
        self.model.infer_end()

        # # debug: save all hyps to a text file
        # with open("hyps.txt", "a") as fp:
        #     if not hasattr(self, "cnt"):
        #         self.cnt = 0
        #     pred_lens = [(t != self.eos).sum(axis=-1) for t in hyps]
        #     pred_sents = [self.tokenizer.detokenize(p[:p_len].tolist()) for p, p_len in zip(hyps, pred_lens)]
        #     print(f"index: {self.cnt:04d}", file=fp)
        #     print('\n'.join(pred_sents) + '\n', file=fp)
        #     self.cnt += 1

        # ctc rescoring
        if ctc_scores is not None:
            scores = scores.view(-1)            # (rns, 1) -> (rns,)
            hyp_lens = self.calc_hyp_lens(hyps)
            ctc_likelihood = self.calc_ctc_likelihood(hyps, hyp_lens, feat_lens, ctc_scores)        # (rns,)
            scores = (1 - self.ctc_beta) * scores + self.ctc_beta * ctc_likelihood

        # lm rescoring
        if self.lm_beta is not None and 0. < self.lm_beta < 1.:                                     # use lm
            assert self.lm is not None
            preds = self.detokenize(hyps[:, 1:])        # [:, 1:]: remove the sos at the head
            lm_scores = torch.tensor([self.lm(pred) for pred in preds], device=device).long()
            scores = (1 - self.lm_beta) * scores + self.lm_beta * lm_scores
        scores = scores.view(-1)    # (rns, 1) -> (rns,)

        # use length bonus. When self.lb_beta is 0., no length bonus is used.
        pred_lens = [(h != self.eos).sum(axis=-1) for h in hyps[:, 1:]]
        if self.lb_beta is not None:
            length_penalty = [((5.0 + (l + 1)) / 6.0) ** self.lb_beta for l in pred_lens]
            length_penalty = torch.tensor(length_penalty, device=device).float().view(-1)    # (rns,)
            scores = scores / length_penalty

        # get the best hyps
        scores = scores.view(-1, bms)  # (rns, bms)
        _, best_hyp_idxs = scores.topk(k=1, dim=-1)  # (bs, k)
        best_hyp_idxs = best_hyp_idxs.view(-1)
        idxs = torch.arange(bs, device=device) * bms
        idxs = idxs.unsqueeze(1).repeat(1, 1).view(-1)
        best_hyp_idxs += idxs
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyp_idxs)

        # return best_hyps
        return best_hyps[:, 1:]  # [:, 1:]: remove the sos at the head

    def calc_hyp_lens(self, hyps):
        rns = hyps.size(0)
        hyp_lens = hyps.new_zeros(rns).long()       # do not use byte tensor, becase its range is [0, 255]
        for i in range(rns):
            if sum(hyps[i] == self.eos) > 0:
                hyp_lens[i] = (hyps[i] == self.eos).nonzero(as_tuple=True)[0][0] + 1  # eos included
            else:  # not terminated
                hyp_lens[i] = hyps.size(1)
        return hyp_lens

    def calc_ctc_likelihood(self, hyps, hyp_lens, feat_lens, ctc_scores):
        bs, seq_len, vocab = ctc_scores.size()
        bms = self.bms
        rns = bs * bms
        assert ctc_scores.size(0) == bs and feat_lens.size(0) == bs and hyps.size(0) == rns
        ctc_scores = ctc_scores.unsqueeze(1).repeat(1, bms, 1, 1).view(rns, seq_len, vocab)
        feat_lens = feat_lens.unsqueeze(1).repeat(1, bms).view(rns)

        ctc_scores = ctc_scores.transpose(0, 1).log_softmax(2)
        # Don't forget the negative sign
        # hyps[:, 1:], hyp_lens - 1: removing the leading sos.
        ctc_likelihood = -F.ctc_loss(
            ctc_scores, hyps[:, 1:], feat_lens, hyp_lens - 1, blank=self.blank, reduction="none"
        )
        # (2022.6.20):
        # Problem: Some values in `ctc_likelihood` could be -inf.
        # Modification: Setting `inf` to the minimal value of the ctc scores of hyps
        ctc_likelihood = ctc_likelihood.view(bs, bms)
        for k in range(bs):
            inf_idxs = (ctc_likelihood[k] == -float("inf")).nonzero(as_tuple=True)[0]
            non_inf_idxs = (ctc_likelihood[k] != -float("inf")).nonzero(as_tuple=True)[0]
            # `non_inf_idxs` could be empty(ctc_likelihood[k] are all `-inf`)
            if len(ctc_likelihood[k, non_inf_idxs]) == 0:
                min_score = 0.
            else:
                min_score = ctc_likelihood[k, non_inf_idxs].min()
            for idx in inf_idxs:
                ctc_likelihood[k, idx] = min_score
        ctc_likelihood = ctc_likelihood.view(-1)

        return ctc_likelihood
