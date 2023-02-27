import logging
import torch

from tqdm import tqdm

from .processors import processors_mapping


logger = logging.getLogger(__name__)

class PromptingDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer

        assert mode in ["train", "dev", "test"]
        self.mode = mode

        if args.mapping is not None:
            self.label_list = self.processor.get_labels()
            self.num_labels = len(self.label_list)
            self.label_to_word = eval(self.args.mapping)

        # Load examples
        logger.info("Loading examples for mode %s", mode)

        self.query_examples = self.processor.get_examples(mode=mode)
        self.size = len(self.query_examples)

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if args.always_preprocess:
            self.preprocess()
        else:
            if mode != "train":
                self.preprocess()
            else:
                self.features = None


    def preprocess(self):
        self.features = []
        self.texts = []

        args = self.args

        verbose = True
        for example in tqdm(self.query_examples):
            template = args.template

            features, text = self.convert_fn(
                example=example,
                template=template,
                verbose=verbose,
            )

            self.features.append(features)
            self.texts.append(text)

            verbose=False


    def __getitem__(self, i):
        if self.features is None:
            example = self.query_examples[i]

            template = self.args.template

            features, _ = self.convert_fn(
                example=example,
                template=template,
                verbose=False,
            )
        else:
            features = self.features[i]

        return features


    def __len__(self):
        return self.size


    def get_labels(self):
        return self.label_list

class TruncateDataset(PromptingDataset):
    def preprocess(self):
        self.texts = []

        for example in tqdm(self.query_examples):
            text = self.convert_fn(example)
            self.texts.append(text)


    def convert_fn(self, example):
        max_length = self.args.max_seq_length
        input_ids = self.tokenizer(example['text'], truncation=True, max_length=max_length, add_special_tokens=False).data['input_ids']

        return self.tokenizer.decode(input_ids)


