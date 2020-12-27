import torch
from torch.utils.data import Dataset


class Corpus(Dataset):

    def __init__(self, data_set):
        self._data = data_set

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        """
        return input word and target word
        1. extract maked lm
        2. extract nsp
        return input_id, segment_id, mask, lm_id, nsp_id

        """
        inp, nsp_trgs, lm_trg, lm_posi = self._data[idx][0], self._data[idx][1], \
                                         self._data[idx][2], self._data[idx][3]

        return torch.tensor(inp), torch.tensor(nsp_trgs), torch.tensor(lm_trg), lm_posi


def collate_fn(batch):
    inp, nsp_trgs, lm_trg, lm_posi = zip(*batch)
    pad_inp = torch.nn.utils.rnn.pad_sequence(inp, batch_first=True)
    return pad_inp, nsp_trgs, lm_trg, lm_posi
