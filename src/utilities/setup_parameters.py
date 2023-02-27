from typing import Optional
from dataclasses import dataclass, field

from transformers import GlueDataTrainingArguments as DataTrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    always_preprocess: Optional[bool] = field(
        default=False,
        metadata={"help": "Preprocess despite the dataset mode"}
    )

    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )
