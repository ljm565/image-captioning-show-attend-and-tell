import re
import os
import math
import random
import pandas as pd
from PIL import Image
import skimage.transform
import matplotlib.pyplot as plt

import torch

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
        caption_file = os.path.join(config.coco_dataset.path, 'captions.txt')
        all_pairs = collect_all_pairs(caption_file)
        train_id, validation_id = make_dataset_ids(len(all_pairs), 1000)     # This ID is only used in the training dataset to create the tokenizer vocabulary
        return {'all_pairs': all_pairs, 'train_id': train_id, 'validation_id': validation_id}
    return None


def topk_accuracy(pred, target, k, eos_token_id):
    pred, target = pred.detach().cpu(), target.detach().cpu()

    total_correct = 0
    batch_size = 0
    for i in range(target.size(0)):
        l = target[i].tolist().index(eos_token_id)
        _, idx = torch.topk(pred[i, :l], k, dim=1)
        correct = idx.eq(target[i, :l].unsqueeze(1).expand_as(idx))
        total_correct += correct.view(-1).float().sum()
        batch_size += l
    return total_correct.item() * (100.0 / batch_size)


def save_figures(img_id, gt, pred, save_path):
    os.makedirs(save_path, exist_ok=True)

    for i, (img, g, p) in enumerate(zip(img_id, gt, pred)):
        plt.figure()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        img = plt.imread(img)
        h, w, _ = img.shape

        plt.imshow(img)
        plt.text(w/2, h+20, 'gt: '+g, horizontalalignment='center')
        plt.text(w/2, h+40, 'pred: '+p, horizontalalignment='center')
        plt.savefig(os.path.join(save_path, 'result_' + str(i+1) + '.jpg'))


def save_attn_figures(img_id, attn_img, pred, save_path, trans, enc_hidden_size):
    os.makedirs(save_path, exist_ok=True)
    mybox={'facecolor':'w','boxstyle':'square','alpha':1}

    for i, (img, attn, p) in enumerate(zip(img_id, attn_img, pred)):
        img = trans(Image.open(img)).permute(1, 2, 0)
        attn = attn.view(enc_hidden_size, enc_hidden_size, -1)
        p = p.split()

        plt.figure(figsize=(4*3, math.ceil(len(p)/4)*3))
        for j in range(attn.size(-1)):
            score = skimage.transform.pyramid_expand(attn[:, :, j].numpy(), upscale=18, sigma=8)
            plt.subplot(math.ceil(len(p)/4), 4, j+1)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.imshow(img)
            plt.imshow(score, cmap='gray', alpha=0.7)
            plt.text(5, 15, p[j], bbox=mybox)
        plt.savefig(os.path.join(save_path, 'result_attn_' + str(i+1) + '.jpg'))
