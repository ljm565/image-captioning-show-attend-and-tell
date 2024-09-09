# Bahdanau Attention Visualization
Here, we provide a guide for visualizing the word-level attention weights of a trained image captioning model.
This method is valid only for models trained using the attention mechanism.

### 1. Attention Visualization
#### 1.1 Arguments
There are several arguments for running `src/run/vis_attention.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to visualize attention. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to visualize.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metric such as BLEU, NIST, etc.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `validation`) Choose one of [`train`, `validation`, `test`].
* [`-n`, `--result_num`]: (default: `10`) The number of random data to visualize.


#### 1.2 Command
`src/run/vis_attention.py` file is used to visualize attention of the model with the following command:
```bash
python3 src/run/vis_attention.py --resume_model_dir ${project}/${name}
```

The attention-related images will be saved at `${project}/${name}/vis_outputs/vis_attention` folder.