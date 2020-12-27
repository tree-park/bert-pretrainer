from enum import Enum

from konlpy.tag import Kkma


class TokenMarks(Enum):
    PAD = '[PAD]'  # 0
    UNKNOWN = '[UKN]'  # 1
    END = '[END]'  # 2
    START = '[SRT]'  # 3
    CLS = '[CLS]'    # 4 Classification token
    SEP = '[SEP]'  # 5
    MASK = '[MASK]'   # 6


class Vocab:
    WORD2IDX = {mark: idx for idx, mark in enumerate(list(TokenMarks), start=0)}

    def __init__(self, min_cnt, need_sep=False):
        self.min_cnt = min_cnt
        self.excepts = '.#$%^&*'
        self.__word2idx = {k: v for k, v in self.WORD2IDX.items()}
        self.__idx2word = {}
        self._idx2word = {}
        if not need_sep:
            del self.__word2idx[TokenMarks.SEP]

    def load(self, corpus: list):
        vocabs = {}
        for sent in corpus:
            for word in sent[1: -1]:
                if word not in vocabs.keys():
                    vocabs[word] = 0
                vocabs[word] += 1
        idx = len(self.WORD2IDX)
        for w in vocabs.keys():
            if self._vocabs_filter(w, vocabs[w]):
                self.__word2idx[w] = idx
                idx += 1

        return self.__word2idx

    def _vocabs_filter(self, v, cnt):
        if cnt < self.min_cnt:
            return
        if v in self.excepts:
            return
        return v

    def to_idx2word(self):
        self.__idx2word = {idx: w for w, idx in self.__word2idx.items()}

    def get_word(self, idx):
        if not self.__idx2word:
            return -1
        return self.__idx2word[idx]

    @property
    def keys(self):
        return self.__word2idx.keys()

    @property
    def word2idx(self):
        return self.__word2idx

    @property
    def idx2word(self):
        return self.__idx2word

    @property
    def vocab(self):
        return self.__word2idx.keys()[len(TokenMarks):]

    def __repr__(self):
        return self.__word2idx

    def __len__(self):
        return len(self.__word2idx)

    def __getitem__(self, item):
        try:
            return self.__word2idx[item]
        except KeyError:
            return self.__word2idx[TokenMarks.UNKNOWN]


def preprocessor(corpus: list, lang='ko'):
    result = []
    tkner = Tokenizer(lang)
    for line in corpus:
        line = line.strip()
        sents = tkner.sent_seperator(line)
        sents = [tkner.tokenizer(sent) for sent in sents]
        sents = [_to_word(sent) for sent in sents]
        for sent in sents:
            if len(sent) < 5:
                continue
            words = [TokenMarks.START] + sent + [TokenMarks.END]
            result.append(words)
    return result


def _to_word(sent: list) -> list:
    """ Filter word as stop words """
    rst = []
    for word in sent:
        word = word.strip()
        if not word or word in '"\'\\₩':
            continue
        rst.append(word)
    return rst


class Tokenizer:
    def __init__(self, lang='ko'):
        if lang == 'ko':
            kkm = Kkma()
            self.sent_seperator = kkm.sentences
            self.tokenizer = kkm.morphs

        else:
            raise
