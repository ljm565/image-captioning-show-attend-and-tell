# Training Transformer Translator
Here, we provide guides for training an image captioning model.

### 1. Configuration Preparation
To train a model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 999
deterministic: True

# environment config
device: [0]     # examples: [0], [0,1], [1,2,3], cpu, mps... 

# project config
project: outputs/image_captioning
name: flickr8k

# model config
img_size: 256
enc_hidden_dim: 14
dec_hidden_dim: 512
dec_num_layers: 1
max_len: 64                         # Maximum length setting of captions
dropout: 0.1
vocab_size: 10000                   # Vocabulary size
using_attention: True

# data config
workers: 0                          # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
flickr8k_train: True                # If True, flickr8k data will be used
flickr8k_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 50
steps: 30000
warmup_steps: 200
enc_lr0: 1e-4
enc_lrf: 0.1                          # enc_last_lr = enc_lr0 * enc_lrf
dec_lr0: 4e-4
dec_lrf: 0.1                          # dec_last_lr = dec_lr0 * dec_lrf
scheduler_type: 'cosine'              # ['linear', 'cosine']
regularization_lambda: 1
patience: 10                          # Early stopping epochs.
topk: 5                               # Logit's top-k accuracy.
prediction_print_n: 10                # Number of examples to show during inference.

# logging config
common: ['train_loss', 'validation_loss', 'enc_lr', 'dec_lr']
metrics: ['ppl', 'bleu2', 'bleu4', 'nist2', 'nist4', 'topk_acc']   # You can add more metrics after implements metric validation codes
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metrics.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```

When training started, the learning rate curve will be saved in `${project}/${name}/vis_outputs` automatically based on the values set in `config/config.yaml`.
When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.