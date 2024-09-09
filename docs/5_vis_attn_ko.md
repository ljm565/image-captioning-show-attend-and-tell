# Bahdanau Attention Visualization
여기서는 학습된 GRU 기계 번역 모델의 단어별 attention 비율을 가시화하는 가이드를 제공합니다.
이 방법은 attention 기법을 사용하여 학습한 모델에 한에서 유효합니다.

### 1. Attention Visualization
#### 1.1 Arguments
`src/run/vis_attention.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-r`, `--resume_model_dir`]: Attention 가시화를 할 모델의 경로. `${project}/${name}`까지의 경로만 입력하면, 자동으로 `${project}/${name}/weights/`의 모델을 선택하여 모델을 로드합니다.
* [`-l`, `--load_model_type`]: [`metric`, `loss`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation metric (e.g. BLEU, NIST, etc.) 최대일 때 모델을 resume.
    * `loss`: Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-d`, `--dataset_type`]: (default: `validation`) [`train`, `validation`, `test`] 중 하나를 선택.
* [`-n`, `--result_num`]: (default: `10`) 가시화할 랜덤 데이터 개수.


#### 1.2 Command
`src/run/vis_attention.py` 파일로 다음과 같은 명령어를 통해 학습된 모델의 단어별 attention을 가시화 합니다.
```bash
python3 src/run/vis_attention.py --resume_model_dir ${project}/${name}
```

Attention 관련 이미지들은 `${project}/${name}/vis_outputs/vis_attention` 폴더에 저장됩니다.