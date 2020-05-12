import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from util import transform_letter_to_index


class load_dataset(Dataset):
    def __init__(self, args, name='train'):
        self.name = name
        if name == 'train':
            speech_path = args._data_root + 'train_new.npy'
            transcript_path = args._data_root + 'train_transcripts.npy'
        elif name == 'val':
            speech_path = args._data_root + 'dev_new.npy'
            transcript_path = args._data_root + 'dev_transcripts.npy'
        elif name == 'test':
            speech_path = args._data_root + 'test_new.npy'

        self.speech = np.load(speech_path, allow_pickle=True, encoding='bytes')
        if self.name != 'test':
            transcript = np.load(transcript_path, allow_pickle=True, encoding='bytes')
            self.transcript = transform_letter_to_index(transcript)

    def __getitem__(self, index):

        if self.name != 'test':
            return torch.tensor(self.speech[index]), torch.LongTensor(self.transcript[index])
        else:
            return torch.tensor(self.speech[index])

    def __len__(self):
        return len(self.speech)


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    # Get the length before padding
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    # Pad the sequence of x and y with 0
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(x_lens), yy_pad, torch.tensor(y_lens)


def pad_collate_test(batch):
    x_lens = [len(x) for x in batch]
    xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
    return xx_pad, torch.tensor(x_lens)


def get_loader(args, name, shuffle=True, drop_last=False):
    dataset = load_dataset(args, name)
    if name == 'test':
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4,
                                collate_fn=pad_collate_test, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4,
                                collate_fn=pad_collate, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
    return dataloader