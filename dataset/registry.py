from typing import Dict, Type, Any

# Import dataset classes
from .xdg.dataset import XDGDataset
from .bigmath.dataset import BigMathDataset
from .math500.dataset import Math500Dataset
from .aime.dataset import AIMEDataset
from .coco.dataset import COCODataset
from .cifar.dataset import CifarDataset
from .instruct_svg.dataset import InstructSVGDataset
from .sgp_bench.dataset import SGPBenchDataset
from .draw_svg.dataset import DrawSVGDataset
from .simple_object.dataset import SimpleObjectDataset
from .simple_relation.dataset import SimpleRelationDataset
from .coco_image.dataset import COCOImageDataset
from .simplelr.dataset import SimplelrDataset
from .haoquan_svg.dataset import HQSVGDataset
from .haoquan_svg_new.dataset import NewHQSVGDataset
from .coco_mix.dataset import MixedSVGImageDataset
from .vgbench.dataset import VGBenchDataset
from .reward import Reward, SVGReward, SVGImageReward, SVGRawImageReward

# Registry for datasets
DATASETS = {
    "xdg": XDGDataset,
    "bigmath": BigMathDataset,
    "math500": Math500Dataset,
    "aime": AIMEDataset,
    "coco": COCODataset,
    "cifar": CifarDataset,
    "instruct_svg": InstructSVGDataset,
    "sgp_bench":SGPBenchDataset,
    "draw_svg": DrawSVGDataset,
    "simple_object": SimpleObjectDataset,
    "simple_relation": SimpleRelationDataset,
    "coco_image": COCOImageDataset,
    "simplelr": SimplelrDataset,
    "hq_svg": HQSVGDataset,
    "hq_svg_new": NewHQSVGDataset,
    "coco_mix": MixedSVGImageDataset,
    "vgbench/VGen": VGBenchDataset,
    # Add more datasets here
}

REWARDS = {
    "xdg": Reward,
    "bigmath": Reward,
    "math500": Reward,
    "aime": Reward,
    "simplelr": Reward,
    "svg": SVGReward,
    "svg_image": SVGImageReward,
    "svg_raw_image": SVGRawImageReward
}



def get_dataset_class(dataset_name: str):
    """Get dataset class by name"""
    if "xiaodonggua" in dataset_name:
        dataset_name = "xdg"
    elif dataset_name == "SynthLabsAI/Big-Math-RL-Verified":
        dataset_name = "bigmath"
    elif dataset_name == "simplelr":
        dataset_name = "simplelr"
    elif dataset_name == "HuggingFaceH4/aime_2024":
        dataset_name = "aime"
    elif dataset_name == "HuggingFaceH4/math_500":
        dataset_name = "math500"
    elif dataset_name == "phiyodr/coco2017":
        dataset_name = "coco"
    elif dataset_name == "uoft-cs/cifar100":
        dataset_name = "cifar"
    elif dataset_name == "uwunion/instruct_svg":
        dataset_name = "instruct_svg"
    elif dataset_name == "sgp-bench/sit_10k":
        dataset_name = "sgp_bench"
    elif dataset_name == "achang/draw_svg":
        dataset_name = "draw_svg"
    elif dataset_name == "simple_object":
        dataset_name = "simple_object"
    elif dataset_name == "simple_relation":
        dataset_name =  "simple_relation"
    elif dataset_name == "HuggingFaceM4/COCO":
        dataset_name = "coco_image"
        
        
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

def get_reward_class(reward_name: str) -> Type[Any]:
    """Get reward class by name"""
    if "xiaodonggua" in reward_name:
        reward_name = "xdg"
    elif reward_name == "SynthLabsAI/Big-Math-RL-Verified":
        reward_name = "bigmath"
    elif reward_name == "simplelr":
        reward_name = "simplelr"
    elif reward_name == "HuggingFaceH4/aime_2024":
        reward_name = "aime"
    elif reward_name == "HuggingFaceH4/math_500":
        reward_name = "math500"
    elif reward_name == "phiyodr/coco2017":
        reward_name = "coco"
    elif reward_name == "uoft-cs/cifar100":
        reward_name = "svg"
    elif reward_name == "uwunion/instruct_svg":
        reward_name = "svg"
        
    elif reward_name == "sgp-bench/sit_10k":
        reward_name = "svg"
    elif reward_name == "achang/draw_svg":
        reward_name = "svg"
    elif reward_name == "simple_object":
        reward_name = "svg"
    elif reward_name == "simple_relation":
        reward_name = "svg"
    elif reward_name == "HuggingFaceM4/COCO":
        reward_name = "svg_raw_image"
    if reward_name not in REWARDS:
        raise ValueError(f"Reward {reward_name} not found. Available rewards: {list(REWARDS.keys())}")
    return REWARDS[reward_name]