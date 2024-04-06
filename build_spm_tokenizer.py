import os
from argparse import ArgumentParser

import sentencepiece as spm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--txt_file_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--sos", type=int, required=True)
    parser.add_argument("--eos", type=int, required=True)
    parser.add_argument("--unk", type=int, required=True)
    parser.add_argument("--norm", type=str, default="identity", help="identity by default (no normalization)")
    parser.add_argument("--unk_str", type=str, default=chr(ord('a') + 72))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(' '.join([
        f"--input={args.txt_file_path} ",
        f"--model_prefix={args.model_prefix}",
        f"--vocab_size={args.vocab_size}",
        f"--model_type={args.model_type}",
        f"--normalization_rule_name={args.norm}",
        f"--control_symbols=<blank>",                                           # for CTC loss
        f"--bos_id={args.sos} --eos_id={args.eos} --unk_id={args.unk}",         # we don't need to set `pad_id` since it's -1 by default
        f"--pad_piece=<ig> --bos_piece=<sos> --eos_piece=<eos> --unk_piece={args.unk_str}",
    ]))
