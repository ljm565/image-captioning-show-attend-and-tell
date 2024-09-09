# Training Transformer Translator
여기서는 이미지 캡셔닝 모델을 학습하는 가이드를 제공합니다.

### 1. Configuration Preparation
Transformer 기계 번역 모델을 학습하기 위해서는 Configuration을 작성하여야 합니다.
Configuration에 대한 option들의 자세한 설명 및 예시는 다음과 같습니다.

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
`src/run/train.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-c`, `--config`]: 학습 수행을 위한 config file 경로.
* [`-m`, `--mode`]: [`train`, `resume`] 중 하나를 선택.
* [`-r`, `--resume_model_dir`]: mode가 `resume`일 때 모델 경로. `${project}/${name}`까지의 경로만 입력하면, 자동으로 `${project}/${name}/weights/`의 모델을 선택하여 resume을 수행.
* [`-l`, `--load_model_type`]: [`metric`, `loss`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation metric (e.g. BLEU, NIST, etc.) 최대일 때 모델을 resume.
    * `loss`: Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-p`, `--port`]: (default: `10001`) DDP 학습 시 NCCL port.


#### 2.2 Command
`src/run/train.py` 파일로 다음과 같은 명령어를 통해 모델을 학습합니다.
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```
학습이 시작되면 예상 학습 learning curve 그래프가 `${project}/${name}/vis_outputs` 경로에 저장됩니다.
모델 학습이 끝나면 `${project}/${name}/weights`에 체크포인트가 저장되며, `${project}/${name}/args.yaml`에 학습 config가 저장됩니다.