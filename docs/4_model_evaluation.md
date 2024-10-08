# Trained Model Evaluation
Here, we provide guides for evaluating the trained image captioning model.


### 1. Evaluation
#### 1.1 Arguments
There are several arguments for running `src/run/validation.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to evaluate. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to evaluate.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's metric such as BLEU, NIST, etc.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `validation`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/validation.py` file is used to evaluate the model with the following command:
```bash
python3 src/run/validation.py --resume_model_dir ${project}/${name}
```