# Setup
## Environment
```bash
pip install uv
uv sync
```

## Data Preparation
Nothing to do.
This is because in `train.py:get_streaming_dataset` we implement a default setting of loading streaming dataset from `DKYoon/SlimPajama-6B`.

# Training

```bash
bash train.sh arm_700m # bdm_700m, mdm_700m, udm_700m
```

# Evaluation Harness
```bash
MODEL_PAHT=<your_checkpoint> bash eval.sh # "ar", "mdm", "udm", "bdm" shoud be contained in the path to distinguish
```


## Acknowledgments
Thanks [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and [LLaDA](https://github.com/ML-GSAI/LLaDA)
for their great work!