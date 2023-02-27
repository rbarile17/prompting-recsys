import logging
import random

from .dataset import PromptingDataset
from .processors import DbbookProcessor, DbbookRankingProcessor

logger = logging.getLogger(__name__)

def tokenize_multipart_input(
    example,
    tokenizer,
    template=None,
    return_tensors=None,
):
    assert template is not None

    template = template.replace('*user_id*', str(example['user_id']))
    template = template.replace('*item_id*', str(example['item_id']))
    template = template.replace('*item_text*', example['item_text'])

    if (example['user_genres'] != ''):
        template = template.replace('*user_genres*', example['user_genres'])
    else:
        template = template.replace(
            'User likes the book genres *user_genres*', 
            'We don\'t know which genres the user likes')
    if return_tensors is not None:
        return tokenizer(template, return_tensors=return_tensors).data

    return tokenizer(template).data


class T5PromptingDataset(PromptingDataset):
    def convert_fn(
        self,
        example,
        template=None,
        verbose=False
    ):
        inputs = tokenize_multipart_input(
            example=example,
            tokenizer=self.tokenizer,
            template=template,
        )

        inputs["labels"] = self.tokenizer(f"{self.label_to_word[example['label']]}").input_ids

        if verbose:
            logger.info("*** Example ***")
            logger.info("text: %s" % self.tokenizer.decode(inputs["input_ids"]))

        return inputs, None


class T5EvalAsRankDataset(PromptingDataset):
    def __init__(self, args, tokenizer, user_id=None):
        self.args = args
        self.task_name = args.task_name
        self.processor = DbbookRankingProcessor("dbbook_ranking")
        self.tokenizer = tokenizer

        self.query_examples = self.processor.get_examples(user_id=user_id)
        self.size = len(self.query_examples)

        self.features = None

    def __getitem__(self, i):
        example = self.query_examples[i]

        template = self.args.template

        features, _ = self.convert_fn(
            example=example,
            template=template,
            verbose=False,
        )

        features["item_id"] = example["item_id"]

        return features

    def convert_fn(
        self,
        example,
        template=None,
        verbose=False
    ):
        inputs = tokenize_multipart_input(
            example=example,
            tokenizer=self.tokenizer,
            template=template,
            return_tensors='pt'
        )

        inputs["input_ids"] = inputs["input_ids"].squeeze()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze()

        if verbose:
            logger.info("*** Example ***")
            logger.info("text: %s" % self.tokenizer.decode(inputs["input_ids"].squeeze().tolist()))

        return inputs, None


class T5PromptingStructuredDataset(PromptingDataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.task_name = args.task_name
        self.processor = DbbookProcessor(args.task_name)
        self.tokenizer = tokenizer

        assert mode in ["train", "dev", "test"]
        self.mode = mode

        if args.mapping is not None:
            self.label_list = self.processor.get_labels()
            self.num_labels = len(self.label_list)
            self.label_to_word = eval(self.args.mapping)

        # Load examples
        logger.info("Loading examples for mode %s", mode)

        self.query_examples = self.processor.get_structured_examples(mode=mode)
        self.size = len(self.query_examples)

        if mode == 'dev':
            self.preprocess()
        else:
            self.features = None


    def tokenize_multipart_input(
        self,
        example,
        template=None
    ):
        assert template is not None

        template = template.replace('*user_id*', str(example['user_id']))
        template = template.replace('*item_id*', str(example['item_id']))
        template = template.replace('*item_genre*', example['item_genre'])
        template = template.replace('*item_author*', example['item_author'])
        template = template.replace('*item_series*', example['item_series'])
        template = template.replace('*item_publisher*', example['item_publisher'])
        template = template.replace('*item_subject*', example['item_subject'])

        if (example['user_genres'] != ''):
            template = template.replace('*user_genres*', example['user_genres'])
        else:
            template = template.replace(
                'User likes the book genres *user_genres*', 
                'We don\'t know which genres the user likes')
        
        return self.tokenizer(template).data


    def convert_fn(
        self,
        example,
        template=None,
        verbose=False
    ):
        inputs = self.tokenize_multipart_input(
            example=example,
            template=template,
        )

        inputs["labels"] = self.tokenizer(f"{self.label_to_word[example['label']]}").input_ids

        if verbose:
            logger.info("*** Example ***")
            logger.info("text: %s" % self.tokenizer.decode(inputs["input_ids"]))

        return inputs, None


class T5EvalAsRankStructuredDataset(PromptingDataset):
    def __init__(self, args, tokenizer, user_id=None):
        self.args = args
        self.task_name = args.task_name
        self.processor = DbbookRankingProcessor("dbbook_ranking")
        self.tokenizer = tokenizer

        self.query_examples = self.processor.get_structured_examples(user_id=user_id)
        self.size = len(self.query_examples)

        self.features = None


    def __getitem__(self, i):
        example = self.query_examples[i]

        template = self.args.template

        features, _ = self.convert_fn(
            example=example,
            template=template,
            verbose=False,
        )

        features["item_id"] = example["item_id"]

        return features


    def tokenize_multipart_input(
        self,
        example,
        template=None
    ):
        assert template is not None

        template = template.replace('*user_id*', str(example['user_id']))
        template = template.replace('*item_id*', str(example['item_id']))
        template = template.replace('*item_genre*', example['item_genre'])
        template = template.replace('*item_author*', example['item_author'])
        template = template.replace('*item_series*', example['item_series'])
        template = template.replace('*item_publisher*', example['item_publisher'])
        template = template.replace('*item_subject*', example['item_subject'])

        if (example['user_genres'] != ''):
            template = template.replace('*user_genres*', example['user_genres'])
        else:
            template = template.replace(
                'User likes the book genres *user_genres*', 
                'We don\'t know which genres the user likes')
        
        return self.tokenizer(template, return_tensors='pt').data


    def convert_fn(
        self,
        example,
        template=None,
        verbose=False
    ):
        inputs = self.tokenize_multipart_input(
            example=example,
            template=template,
        )

        inputs["input_ids"] = inputs["input_ids"].squeeze()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze()

        if verbose:
            logger.info("*** Example ***")
            logger.info("text: %s" % self.tokenizer.decode(inputs["input_ids"].squeeze().tolist()))

        return inputs, None