
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, distributed

from models import ResNetEncoder, LSTMDecoder
from tools.tokenizers import CustomTokenizer
from utils import LOGGER, RANK, colorstr
from utils.filesys_utils import read_dataset
from utils.data_utils import DLoader, seed_worker
from utils.func_utils import *

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_tokenizers(config, preparation):
    if config.coco_train:
        tokenizer = CustomTokenizer(config, preparation['all_pairs'], preparation['train_id'])
    else:
        LOGGER.warning(colorstr('yellow', 'You must implement your custom tokenizer loading codes..'))
        raise NotImplementedError
    return tokenizer


def get_model(config, tokenizer, device):
    encoder = ResNetEncoder(config).to(device)
    decoder = LSTMDecoder(config, tokenizer).to(device)
    return encoder, decoder


def build_dataset(config, tokenizer, modes, preparation):
    if config.coco_train:
        # for the image preprocessing
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = transforms.Compose([transforms.Resize((config.img_size, config.img_size)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
                                
        img_folder = os.path.join(config.coco_dataset.path, 'images')
        all_pairs = preparation['all_pairs']
        config.img_folder = img_folder
        
        dataset_dict = {mode: DLoader(img_folder, all_pairs, trans, preparation[f'{mode}_id'], tokenizer, config.max_len) for mode in modes if f'{mode}_id' in preparation}
    else:
        LOGGER.warning(colorstr('yellow', 'You have to implement data pre-processing code..'))
        # dataset_dict = {mode: CustomDLoader(config.CUSTOM.get(f'{mode}_data_path')) for mode in modes}
        raise NotImplementedError
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, tokenizer, modes, preparation, is_ddp=False):
    datasets = build_dataset(config, tokenizer, modes, preparation)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes if m in datasets}

    return dataloaders
