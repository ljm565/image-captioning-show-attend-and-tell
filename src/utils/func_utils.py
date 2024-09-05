import re
import os
import random
import pandas as pd

from utils import LOGGER, colorstr



def print_samples(target, prediction):
    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr('GT        : ') + target)
    LOGGER.info(colorstr('Prediction: ') + prediction)
    LOGGER.info('-'*100 + '\n')


def collect_all_pairs(caption_file):
    captions = pd.read_csv(caption_file)
    all_pairs = [[img, preprocessing(cap.lower())] for img, cap in zip(captions['image'], captions['caption'])]
    return all_pairs
    

def make_dataset_ids(total_l, valset_l):
    all_id = list(range(total_l))
    trainset_id = random.sample(all_id, total_l - valset_l)
    valset_id = list(set(all_id) - set(trainset_id))
    return trainset_id, valset_id


def preprocessing(s):
    s = re.sub('[#$%&()*+\-/:;<=>@\[\]^_`{|}~\'".,?!]', '', s).lower()
    s = ' '.join(s.split())
    return s


def prepare_necessary(config):
    if config.coco_dataset:
        caption_file = os.path.join(config.coco_dataset.path, 'cpations.txt')
        all_pairs = collect_all_pairs(caption_file)
        train_id, validation_id = make_dataset_ids(len(all_pairs), 1000)     # This ID is only used in the training dataset to create the tokenizer vocabulary
        return {'all_pairs': all_pairs, 'train_id': train_id, 'validation_id': validation_id}
    return None
