import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset



def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DLoader(Dataset):
    def __init__(self, img_folder, all_pairs, trans, ids, tokenizer, max_len):
        self.data_pairs = [all_pairs[id] for id in ids]
        self.trans = trans
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [[self.trans(Image.open(img_folder+img_name)), cap] for img_name, cap in tqdm(self.data_pairs, desc='load data')]
        self.length = len(self.data)


    def __getitem__(self, idx):
        image, caption = self.data[idx][0], self.data[idx][1]
        caption = [self.tokenizer.bos_token_id] + self.tokenizer.encode(caption)[:self.max_len-2] + [self.tokenizer.eos_token_id]
        caption = caption + [self.tokenizer.pad_token_id] * (self.max_len - len(caption))
        return image, torch.LongTensor(caption), self.ids[idx]


    def __len__(self):
        return self.length