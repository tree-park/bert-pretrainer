import torch
from torch.utils.data import Dataset


class Corpus(Dataset):

    def __init__(self, data_set):
        self._data = data_set

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        """ return input word and target word """
        ko, en = self._data[idx]
        return torch.tensor(ko), torch.tensor(en)


def collate_fn(batch):
    pad = torch.nn.utils.rnn.pad_sequence(batch,batch_first=True)
    return pad
