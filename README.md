

# Symbolic Graphics Programming with Large Language Models

<div align="center">
  <img src="./assets/teaser.png" alt="teaser image" width="820">
  <br>
  <sup><em>As training progresses, we can observe that the model acquires better
compositional drawing ability, producing semantically accurate symbolic graphics programs.</em></sup>
</div>

## Introduction

LLMs are strong at coding, but their ability to write symbolic graphics programs (SGPs) that render images (especially SVGs) is underexplored. This work studies text-to-SGP generation as a probe of visual generation, introducing SGP-GenBench to evaluate object-, scene-, and composition-level performance across open and proprietary models, revealing notable shortcomings. To improve results, we use reinforcement learning with rewards from visual–text similarity scores, which steadily enhances SVG quality and semantic alignment. Experiments show substantial gains, bringing performance close to state-of-the-art closed-source models.

## Installation

```bash
conda env create -n sgp_gen -f environment.yml

conda activate sgp_gen

pip install vllm==0.7.2 && pip install oat-llm==0.0.9

git clone git@github.com:Sphere-AI-Lab/SGP-RL.git

cd SGP-RL

pip install -e .

pip install cairosvg openai-clip lxml
```
We provide a requirements.txt recording the version informations for reference.


## Quick Start

### Prepare Datasets

#### Automatically Download All Required Data

We provide a script for automatically downloading all training and evaluation datasets:
```bash
bash prepare_data.sh
```

#### Manually Download datasets

Alternatively, you can also download the datasets manually.

##### Training Datasets


1) Setup COCO 2017 dataset(assume you put it in COCO_DIR):
```bash
export COCO_DIR=YOUR_COCO_DIR
cd "$COCO_DIR"
# Images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Captions annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

```
You should have:
- train2017/
- val2017/
- annotations/
    - captions_train2017.json
    - captions_val2017.json
    ...

2) setup the svg training data:
download the dataset file at https://huggingface.co/datasets/SphereLab/SGP-Gen-70k/resolve/main/svg-gen-70k.jsonl

Put it into YOUR_SVG_DIR and setup environment variables:
```bash
export SVG_DIR=YOUR_SVG_DIR
```


##### SGP-Object Datasets (for evaluation)
Download SGP-Object dataset at  
https://huggingface.co/datasets/SphereLab/SGP-Object/resolve/main/SGP-Object.json

Put it into YOUR_SVG_DIR.


### RL Training

```bash
bash train_zero_svg.sh
```

### Evaluation on SGP-GenBench
Sampling model responses and calculating DINO-score, CLIP-score and Diversity on SGP-:
```bash
# modify the sampling parameters to match your training settings
python evaluate_svg_model.py 
    --model_path YOUR_MODEL_PATH  
```
By default, the sampled responses are stored in ./evaluation_results
To print out the important metrics as csv, 
```bash
# modify the sampling parameters to match your training settings
bash print_results.sh
```
To get the VQA and HPS metrics, check [eval_tools](eval_tools/)

Evaluation on SGP-CompBench:
[sgp-compbench](sgp-compbench/)


## Checkpoint Downloading
We provide a model checkpoint at training step 780 on HuggingFace and Modelscope:

```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="SphereLab/SGP-RL")
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('NOrangeroli/SGP_RL')
```

Download via git:
```bash
git clone https://huggingface.co/SphereLab/SGP-RL
git clone https://www.modelscope.cn/NOrangeroli/SGP_RL.git
```

## Interactive Inference

We provide a script for interactively run the model:

```bash
python interactive_inference.py --model your/model/path
```

Always prompt in the following form:

```bash
Please write SVG code for generating the image corresponding to the following description: YOUR_DESCRIPTION
```

## Acknowledgement
This code is based on [understand-r1-zero](https://github.com/sail-sg/understand-r1-zero).


## Citation

```bib
@article{chen2025symbolic,
  title={Symbolic Graphics Programming with Large Language Models}, 
  author={Yamei Chen and Haoquan Zhang and Yangyi Huang and Zeju Qiu and Kaipeng Zhang and Yandong Wen and Weiyang Liu},
  journal={arXiv preprint arXiv:2509.05208},
  year={2025}
}
```

