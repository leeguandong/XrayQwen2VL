---
base_model: /home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2-VL-7B-Instruct/
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [/home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2-VL-7B-Instruct/](https://huggingface.co//home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2-VL-7B-Instruct/) on the xray dataset.
It achieves the following results on the evaluation set:
- Loss: 1.7299

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 256
- total_eval_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.45.0.dev0
- Pytorch 2.4.1+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1