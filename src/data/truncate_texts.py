import pandas as pd

import yaml

from .paths import PROCESSED_DATA_PATH

from transformers import AutoTokenizer
from transformers import HfArgumentParser

from ..utilities.setup_parameters import DynamicDataTrainingArguments
from .dataset import TruncateDataset

def main(data_args: DynamicDataTrainingArguments):
    OUTPUT_PATH = PROCESSED_DATA_PATH / "dbbook"

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

    dataset = (
        TruncateDataset(data_args, tokenizer=tokenizer, mode="train")
    )

    dataset.texts = [
        (int(text.split(";;")[0]), str(text.split(";;")[1])) 
        for text in dataset.texts
    ]
    texts_df = pd.DataFrame(dataset.texts, columns=["item_id", "item_text"])

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "texts").mkdir(parents=True, exist_ok=True)

    # save each row of texts df as a txt file in the processed data directory
    for _, row in texts_df.iterrows():
        item_id = row["item_id"]
        item_text = row["item_text"]
        with open(OUTPUT_PATH / "texts" / f"{item_id}.text", "w") as f:
            f.write(item_text)

if __name__ == '__main__':
    params = yaml.safe_load(open("params.yaml"))["truncate_texts"]

    parser = HfArgumentParser((DynamicDataTrainingArguments))
    data_args = parser.parse_dict(params)

    main(data_args[0])