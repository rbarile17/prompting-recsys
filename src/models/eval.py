import logging
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd

from typing import Callable, Dict
from tqdm import tqdm

from transformers import (
    set_seed, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, 
    HfArgumentParser, Seq2SeqTrainingArguments, EvalPrediction,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader
from ..data.t5dataset import T5EvalAsRankDataset

from ..data.processors import num_labels_mapping, output_modes_mapping, compute_metrics_mapping
from ..utilities.setup_parameters import ModelArguments, DynamicDataTrainingArguments

logger = logging.getLogger(__name__)


# Build metric
def build_compute_metrics_fn(task_name, tokenizer) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        predictions = p.predictions
        labels = p.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        decoded_preds = tokenizer.batch_decode(predictions)

        decoded_preds = [
            pred.split('<pad>')[1].split('</s>')[0].strip()
            for pred in decoded_preds
        ]

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        return compute_metrics_mapping[task_name](task_name, np.array(decoded_preds), np.array(decoded_labels))

    return compute_metrics_fn


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, Seq2SeqTrainingArguments))

    params_files = [
        ('./params/flan-t5-base-dbbook-prompt-4.json', 4965),
    ]

    for params_file, checkpoint in params_files:
        model_args, data_args, training_args = parser.parse_json_file(params_file)
        data_args.task_name = "dbbook_ranking"
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
            logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
        except KeyError:
            raise ValueError("Task not found: %s" % (data_args.task_name))

        # Create config
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
        )

        set_seed(training_args.seed)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            training_args.output_dir + f'/checkpoint-{checkpoint}',
            config=config
        )
        # Pass dataset and argument information to the model
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

        test_df = pd.read_csv(
            './data/raw/dbbook/test.tsv', 
            sep='\t', header=None, names=['user_id', 'item_id', 'label'])
        user_id_list = test_df.user_id.unique().tolist()

        for user_id in tqdm(user_id_list):
            user_dataset = (T5EvalAsRankDataset(data_args, tokenizer=tokenizer, user_id=user_id))
            user_dataloader = DataLoader(
                user_dataset, 
                batch_size=training_args.per_device_eval_batch_size, 
                shuffle=False,
                collate_fn=data_collator)
            
            scores = {}
            for batch in user_dataloader:
                output = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=3, 
                    output_scores=True, return_dict_in_generate=True)
                scores.update({
                    item_id.item(): score[333].item() 
                    for item_id, score in zip(batch["item_id"], output.scores[0])
                })

            # sort scores by value
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # write to file
            Path('results').mkdir(parents=True, exist_ok=True)
            with open(Path('results') / f"{params_file.split('.')[1].split('/')[2]}.txt", "a") as f:
                for item_id, _ in sorted_scores:
                    f.write(f"{user_id}\t{item_id}\n")
if __name__ == "__main__":
    main()
