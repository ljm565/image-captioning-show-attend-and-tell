# Image Captioning (Show, Attend and Tell)
한국어 버전의 설명은 [여기](./docs/README_ko.md)를 참고하시기 바랍니다.

## Introduction
Here, we are training an image captioning model based on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.
In this code, I have implemented the image captioning model introduced in the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) paper.
For a detailed explanation of this paper, please refer to [Image Captioning (Show, Attend and Tell)](https://ljm565.github.io/contents/img2txt1.html).
The attention mechanism in this code is implemented as the soft attention described in the paper.
For model initialization, I referred to the model initialization method from the well-known PyTorch code for implementing the Show, Attend and Tell paper in the [PyTorch Tutorial to Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
In addition, I found the approach used in the tutorial, where sequences are sorted by length and decoded one by one, to be inefficient. Therefore, I trained the model by decoding all sequences up to the maximum length at once.
Finally, I implemented the soft attention mechanism based on Bahdanau attention using Tanh, as introduced in the original paper (if time permits, I also plan to add hard attention and beam search code).
<br><br><br>

## Supported Models
* Image Encoder: ResNet-101
* Caption Decoder: LSTM
* Attention: Soft Attention (Bahdanau Attention)
<br><br><br>

## Supported Tokenizer
### Custom Word Tokenizer
* Tokenization based on words for attention visualization.
<br><br><br>

## Base Dataset
* Base dataset for this tutorial is [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
The `data_sample` mentioned here is only a subset of the entire dataset.
You can download the full dataset from [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
If you want to check if the code runs properly, you need to execute the following command first to rename the data folder.
    ```bash
    mv data_sample data
    ```
* Custom datasets can also be used by setting the path in the `config/config.yaml`.
However, implementing a custom dataloader may require additional coding work in `src/trainer/build.py`.
<br><br><br>

## Supported Devices
* CPU, GPU, multi-GPU (DDP), MPS (for Mac and torch>=1.12.0)
<br><br><br>

## Quick Start
```bash
python3 src/run/train.py --config config/config.yaml --mode train
```
<br><br>

## Project Tree
This repository is structured as follows.
```
├── configs                           <- Folder for storing config files
│   └── *.yaml
│
└── src      
    ├── models
    |   ├── decoder.py                <- Decoder model file
    |   ├── encoder.py                <- Encoder model file
    |   └── modules.py                <- Modules used in encoder and decoder
    |
    ├── run                   
    |   ├── train.py                  <- Training execution file
    |   ├── validation.py             <- Trained model evaulation execution file
    |   └── vis_attention.py          <- Attention visualization code for each word of attention model
    |
    ├── tools                   
    |   ├── tokenizers
    |   |    └── tokenizer.py         <- Word tokenizer file
    |   |
    |   ├── early_stopper.py          <- Early stopper class file
    |   ├── evaluator.py              <- Metric evaluator class file
    |   ├── model_manager.py          
    |   └── training_logger.py        <- Training logger class file
    |
    ├── trainer                 
    |   ├── build.py                  <- Codes for initializing dataset, dataloader, etc.
    |   └── trainer.py                <- Class for training and evaluating
    |
    └── uitls                   
        ├── __init__.py               <- File for initializing the logger, versioning, etc.
        ├── data_utils.py             <- File defining the custom dataset dataloader
        ├── filesys_utils.py       
        ├── func_utils.py       
        └── training_utils.py     
```
<br><br>


## Tutorials & Documentations
Please follow the steps below to train an image captioning model.
1. [Getting Started](./docs/1_getting_started.md)
2. [Data Preparation](./docs/2_data_preparation.md)
3. [Training](./docs/3_trainig.md)
4. ETC
   * [Evaluation](./docs/4_model_evaluation.md)
   * [Attention Visualization](./docs/5_vis_attn.md)

<br><br><br>

## Training Results
### Image Captioning Training Results
* Loss History<br>
    <img src="docs/figs/img1.png" width="80%"><br><br>

* Validation Set BLEU<br>
    <img src="docs/figs/img2.png" width="80%"><br>
    * Best BLEU-2: 0.3053 (22 epoch)
    * Best BLEU-4: 0.1402 (23 epoch)
    <br><br>

* Validation Set NIST<br>
    <img src="docs/figs/img3.png" width="80%"><br>
    * Best NIST-2: 3.9032 (22 epoch)
    * Best NIST-4: 4.1633 (22 epoch)
    <br><br>

* Validation Set Top-5 Accuracy<br>
    <img src="docs/figs/img4.png" width="80%"><br>
    * Best top-5 accuracy: 73.3850 (16 epoch)
    <br><br>

### Real Examples
For more results, please refer to `figs/image_captioning_model` folder
* Sample 1
<img src="docs/figs/image_captioning_model/result_3.jpg" width="100%"><br>
<img src="docs/figs/image_captioning_model/result_attn_3.jpg" width="100%"><br>
<br><br>

* Sample 2
<img src="docs/figs/image_captioning_model/result_5.jpg" width="100%"><br>
<img src="docs/figs/image_captioning_model/result_attn_5.jpg" width="100%"><br>
<br><br>

* Sample 3
<img src="docs/figs/image_captioning_model/result_6.jpg" width="100%"><br>
<img src="docs/figs/image_captioning_model/result_attn_6.jpg" width="100%"><br>
<br><br>

* Sample 4
<img src="docs/figs/image_captioning_model/result_7.jpg" width="100%"><br>
<img src="docs/figs/image_captioning_model/result_attn_7.jpg" width="100%"><br>
<br><br>

* Sample 5
<img src="docs/figs/image_captioning_model/result_10.jpg" width="100%"><br>
<img src="docs/figs/image_captioning_model/result_attn_10.jpg" width="100%"><br>
<br><br>
        

<br><br><br>
