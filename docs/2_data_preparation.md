# Data Preparation
Here, we will train an image captioning model using [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.
Please refer to the following instructions to utilize custom datasets.


### 1. Flickr8k
If you want to train on the Flickr8k dataset, simply set the `flickr8k_train` value to `True` in the `config/config.yaml` file as follows.
```yaml
flickr8k_train: True                # If True, flickr8k data will be used.
flickr8k_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train on the custom dataset, simply set the `flickr8k_train` value to `False` in the `config/config.yaml` file as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
You have to set your custom training/validation/test datasets.
```yaml
flickr8k_train: False               # If True, flickr8k data will be used.
flickr8k_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>