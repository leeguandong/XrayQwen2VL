## XrayQwen2VL

Xray Large Multi-model Model，基于Qwen2VL微调Xray的多模态大模型，在4张A800上基于qwen2-vl-7b-instruct模型微调。

 <p align="center">
      <a href='https://github.com/leeguandong/XrayQwen2VL'>
            <img src='https://img.shields.io/badge/Project-Page-Green'>
      </a>
      <a href='https://github.com/leeguandong/XrayQwen2VL'>
            <img src='https://img.shields.io/badge/Paper-Arxiv-red'>
      </a>
      </br>
      <a href="https://github.com/leeguandong/XrayQwen2VL/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/leeguandong/XrayQwen2VL" />
      </a>
      <a href="https://github.com/leeguandong/XrayQwen2VL/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/leeguandong/XrayQwen2VL?color=0088ff" />
      </a>
      <a href="https://github.com/leeguandong/XrayQwen2VL/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/leeguandong/XrayQwen2VL?color=0088ff" />
      </a>
      <a href=href="https://github.com/leeguandong/XrayQwen2VL/stargazers">
        <img src="https://img.shields.io/github/stars/leeguandong/XrayQwen2VL?color=ccf">
      </a>
      <a href=href="https://github.com/leeguandong/XrayQwen2VL">
        <img src="https://img.shields.io/github/repo-size/leeguandong/XrayQwen2VL.svg?style=flat-square">
      </a>
      </br>
      <a href=href="https://github.com/leeguandong/XrayQwen2VL">
        <img src="https://visitor-badge.laobi.icu/badge?page_id=https://github.com/leeguandong/XrayQwen2VL">
      </a>
      <a href=href="https://github.com/leeguandong/XrayQwen2VL">
        <img src="https://img.shields.io/github/last-commit/leeguandong/XrayQwen2VL">
      </a>
      <a href="https://github.com/leeguandong/XrayQwen2VL/blob/main/LICENSE">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" />
      </a>
  </p>

## 本文贡献

![](./doc/xrayqwenvl.png)

- 借助Xray开源数据集，基于Qwen2vl训练微调，并开放了用于学术研究的训练lora权重，推理时需要自行加载原始的qwen2-vl-7b-instruct权重，可借助llamafactory中的merge lora进行权重合并。本次使用llamafactory0.9.0进行微调。
## 数据集

- [OpenI](https://openi.nlm.nih.gov/faq#collection)是一个来自印第安纳大学医院的胸部X光片数据集，包括6,459张图像和3,955个报告。

在上述工作中，报告信息都为非结构化的，不利于科学研究。为了生成合理的医学报告，我们对两个数据集进行了预处理，并最终得到了可以用于训练的**英文报告**。除此之外，为了更好的支持中文社区发展，借助ChatGPT的能力，我们将英文报告进行了中文翻译，并最终形成了可用于训练的数据集。

|数据集|数量|下载链接|质量|
|:-|:-|:-|:-|
|OpenI-zh|6,423|[诊疗报告(英文)](./data/openi-en.json)、[诊疗报告(中文)](./data/Xray/openi-zh.json) 、[X光影像](https://pan.baidu.com/s/13GBsDMKf6xBZBSHpoWH_EA?pwd=k9sh)|低|
|OpenI-zh-plus|6,423|-|高|

## 快速上手

### 1.安装环境
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
### 2.模型推理

|模型权重|下载链接|质量|微调方法|
|:-|:-|:-|:-|
|checkpoints-XrayQwen2VL-66|XrayQwen2VL/weights/qwen2_vl-7b/lora/sft|低|LoRA|

#### CLI推理

```python
merge lora

yaml文件
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: "/home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2-VL-7B-Instruct/"
adapter_name_or_path: /home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/saves/qwen2_vl-7b/lora/sft
template: qwen2_vl
finetuning_type: lora

### export
export_dir: /home/lgd/e_commerce_lmm/results/qwen2_vl_lora_sft
export_size: 2
export_device: cpu
export_legacy_format: false

llamafactory-cli export "/home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/examples/merge_lora/qwen2vl_lora_sft.yaml"
```

```python
inference

yaml文件
### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: "/home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2-VL-7B-Instruct/"
adapter_name_or_path: /home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/saves/qwen2_vl-7b/lora/sft
template: qwen2_vl
finetuning_type: lora

llamafactory-cli chat "/home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/examples/inference/qwen2_vl.yaml"
```
### 3.模型训练（复现XrayQwen2VL）

<details>
  <summary>硬件资源</summary>
  <p>* 实验在A800 (4X, 80GB)上进行</p>
</details>
- （1）准备[诊疗报告(中文)](./data/openai-zh-llamafactory-qwen2vl-prompt.json)和[X光影像](https://pan.baidu.com/s/13GBsDMKf6xBZBSHpoWH_EA?pwd=k9sh)在`data/Xray`文件夹下；
- （2）开始训练：
```bash
yaml文件
### model
model_name_or_path: "/home/lgd/common/ComfyUI/models/LLM/qwen/Qwen2-VL-7B-Instruct/"

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
deepspeed: "/home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/examples/deepspeed/ds_z2_config.json"


### dataset
dataset_dir: /home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/data/
dataset: xray  # video: mllm_video_demo
template: qwen2_vl
cutoff_len: 1024
#max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: /home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/saves/qwen2_vl-7b/lora/sft
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 1000

export export CUDA_VISIBLE_DEVICES=3,4,5,6
llamafactory-cli train "/home/lgd/e_commerce_lmm/LLaMA-Factory-0.9.0/examples/train_lora/qwen2vl_lora_sft.yaml"
```
这里的复现过程非常简单，主要是很多过程我们都为大家准备好了，大家可以随时复现一个自己的`XrayQwenVL`。

## 效果展示

*以下效果来自于**低质量**的数据训练和权重

影像诊断：
|X光影像|诊疗报告(医生)|XrayQwen2VL|
|:-|:-|:-|
|![](./data/test/xray/1425_2.png)|患者的胸部X光片显示双侧肺部清晰，没有任何合并症、气胸或胸腔积液的迹象。心脏纵隔的轮廓和胸腔的骨质结构都在正常范围内。X光片的印象表明，没有急性心肺功能异常。这表明病人在这个时候有一个健康的呼吸和心血管系统。|胸部X光片显示，心脏大小正常，纵隔轮廓正常。肺部清晰，没有病灶浸润或气胸的迹象。然而，胸椎有退行性改变。总的来说，X光片显示没有急性心肺异常。|
|![](./data/test/xray/2808_1.png)|X光片显示有陈旧性肉芽肿病的后遗症，但肺部清晰，没有任何局灶性固结、胸腔积液或气胸。心脏和纵膈的大小和轮廓正常，表明没有急性心肺功能异常。总的来说，这是一张清晰的胸部X光片。|X光图像显示，心脏大小正常，肺部清晰，没有局灶性固结、胸腔积液或气胸的迹象。然而，左上叶有稳定的钙化肉芽肿。没有发现急性心肺功能异常。|
|![](./data/test/xray/348_2.png)|研究结果表明，心脏大小和纵隔轮廓正常，肺部清晰，肺血管正常，没有胸腔积液或气孔。此外，观察到胸椎的退行性变化很小。根据印象，没有急性心肺异常的迹象。根据这些结果，病人的整体状况似乎是最佳的。|X光图像显示，心脏大小正常，肺部清晰，没有局灶性固结、胸腔积液或气胸的迹象。然而，左上叶有稳定的钙化肉芽肿。没有发现急性心肺功能异常。|
|![](./data/test/xray/1785_2.png)|胸部X光片显示心脏和纵隔轮廓正常，气管旁区域有肉芽肿性钙化，右上叶有轻度条纹状瘢痕。没有发现活动性肺炎或气胸。骨骼结构完整。根据这些发现，没有大叶性肺炎存在。|X光图像显示，心脏大小正常，肺部清晰，没有局灶性固结、胸腔积液或气胸的迹象。然而，左上叶有稳定的钙化肉芽肿。没有发现急性心肺功能异常。|

## 项目致谢

1. [XrayGLM](https://github.com/THUDM/VisualGLM-6B)为我们提供了数据集；
1. [LLamaFactory](https://github.com/hiyouga/LLaMA-Factory)为我们提供了训练框架；

## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目无法对其准确性作出保证。即使本项目模型输出符合医学事实，也不能被用作实际医学诊断的依据。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

## 使用许可

此存储库遵循[CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/) ，请参阅许可条款。

