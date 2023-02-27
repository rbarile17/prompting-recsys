import logging
import os
import sys

from transformers import (
    set_seed, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, HfArgumentParser, Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from ..data.t5dataset import T5PromptingDataset
from ..data.processors import num_labels_mapping, output_modes_mapping
from ..utilities.setup_parameters import (
    ModelArguments, DynamicDataTrainingArguments
)
from .eval import build_compute_metrics_fn

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, Seq2SeqTrainingArguments))
    for params_file in ["./params/flan-t5-base-dbbook-prompt-4.json"]:
        model_args, data_args, training_args = parser.parse_json_file(params_file)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )

        # Set seed
        set_seed(training_args.seed)

        # Log task info
        try:
            num_labels = num_labels_mapping[data_args.task_name]
            output_mode = output_modes_mapping[data_args.task_name]
            logger.info(
                f"Task name: {data_args.task_name}, number of labels: {num_labels}, output mode: {output_mode}")
        except KeyError:
            raise ValueError(f"Task not found: {data_args.task_name}")

        # Create config
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

        train_dataset = (T5PromptingDataset(data_args, tokenizer=tokenizer, mode="train"))
        eval_dataset = (T5PromptingDataset(data_args, tokenizer=tokenizer, mode="dev"))

        set_seed(training_args.seed)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Pass dataset and argument information to the model
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(eval_dataset.args.task_name, tokenizer)
        )

        trainer.train()

if __name__ == "__main__":
    main()
