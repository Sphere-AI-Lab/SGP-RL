from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import torch
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)

class InstructSVGDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
    

    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from HuggingFace or local source
        """
        dataset = load_dataset(dataset_name)
        dataset['test'] = dataset['train']
        for split in dataset:
            if "solution" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("solution")
            
            dataset[split] = dataset[split].rename_column("output", "svg")
            dataset[split] = dataset[split].rename_column("input", "solution")
        dataset = dataset.map(InstructSVGDataset.process_example)
        # import pdb; pdb.set_trace()
        # for split in dataset:
        #     dataset[split] = dataset[split].filter(
        #         lambda example: 'black and white' not in example['input'].lower()
        #     )
        
        # Apply filtering or selection if needed
        # dataset = dataset['test']
        
        if max_train_samples and max_train_samples > 0:
            for split in dataset:
                if split == "train":
                    dataset[split] = dataset[split].select(range(min(max_train_samples, len(dataset[split]))))
        
        if max_test_samples and max_test_samples > 0:       
            for split in dataset:
                if split == "test":
                    dataset[split] = dataset[split].select(range(min(max_test_samples, len(dataset[split]))))    
        return dataset
    
    @staticmethod
    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
        
        
        return {
            "prompt":  (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
            "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: Please write SVG code for generating the image corresponding to the following description: "
            + example["solution"]
            + "\nAssistant: <think>"
        ),
            # "solution": example["input"],
            # "svg": example["output"]
        }
            
   