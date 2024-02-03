# coding=utf-8
# Contact: bingquanxia@qq.com

class CharTokenizer(object):
    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case
        self.char2idx = {
            "<sos>": 0, "<eos>": 1,
            'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8,
            'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15,
            'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21,
            'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27, ' ': 28
        }
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab = len(self.char2idx)
        self.sos_id = self.char2idx['<sos>']
        self.eos_id = self.char2idx['<eos>']
        self.skipped = set()

    def tokenize(self, text):
        if self.do_lower_case:
            text = text.lower()
        remained = [char for char in text if char in self.char2idx]
        skipped = [char for char in text if char not in self.char2idx]
        if len(skipped) > 0:
            for s in skipped:
                if s not in self.skipped:
                    print(f"Skipped character: {s}")
                    self.skipped.add(s)
        return [self.char2idx[char] for char in remained]

    def detokenize(self, token_ids):
        remained = [d for d in token_ids if d in self.idx2char]
        skipped = [d for d in token_ids if d not in self.idx2char]
        if len(skipped) > 0:
            print(f"Skipped token ids: {skipped}")
        return ''.join([self.idx2char[d] for d in remained])


if __name__ == '__main__':
    tokenizer = CharTokenizer()
    print(tokenizer.tokenize('hello world'))
    # output: [9, 6, 13, 13, 16, 28, 24, 16, 19, 13, 5]
    print(tokenizer.detokenize([9, 6, 13, 13, 16, 28, 24, 16, 19, 13, 5]))
    # output: 'hello world'
