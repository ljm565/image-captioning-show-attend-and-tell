import os
import random
import numpy as np
from PIL import Image

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
        self.img_folder = img_folder
        self.trans = trans
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.length = len(self.data_pairs)


    def __getitem__(self, idx):
        img_name, caption = self.data_pairs[idx]
        image = self.trans(Image.open(os.path.join(self.img_folder, img_name)))
        caption = [self.tokenizer.bos_token_id] + self.tokenizer.encode(caption)[:self.max_len-2] + [self.tokenizer.eos_token_id]
        caption = caption + [self.tokenizer.pad_token_id] * (self.max_len - len(caption))
        return image, torch.tensor(caption, dtype=torch.long), self.ids[idx]


    def __len__(self):
        return self.length