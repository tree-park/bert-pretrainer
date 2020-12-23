from konlpy.tag import Kkma

UNKNOWN = '[UKN]'
CLOSE = '[CLS]'
PAD = '[PAD]'
START = '[SRT]'


class Vocab:
    
    WORD2IDX = {PAD: 0, UNKNOWN: 1, CLOSE: 2, START: 3}

    def __init__(self, min_cnt):
        self.min_cnt = min_cnt
        self.excepts = '.#$%^&*'
        self.word2idx = {k: v for k, v in self.WORD2IDX.items()}
        self.idx2word = {}

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
                self.word2idx[w] = idx
                idx += 1
        
        return self.word2idx

    def _vocabs_filter(self, v, cnt):
        if cnt < self.min_cnt:
            return
        if v in self.excepts:
            return
        return v

    def to_idx2word(self):
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}

    def get_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, item):
        try:
            return self.word2idx[item]
        except KeyError:
            return self.word2idx[UNKNOWN]


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
            words = [START] + sent + [CLOSE]
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


