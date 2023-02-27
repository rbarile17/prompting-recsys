import os
import pandas as pd

from pathlib import Path

from .paths import INTERIM_DATA_PATH, PROCESSED_DATA_PATH


def get_item_text(item_path):
    try:
        with open(item_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

class TruncateProcessor:
    def __init__(self, task_name):
        self.task_name = task_name

    def get_examples(self, data_dir, mode=None):
        files = os.listdir(Path(data_dir) / 'texts')
        files = [Path(data_dir) / 'texts' / file for file in files]
        texts = [f"{file.name.split('.')[0]};;{get_item_text(file)}" for file in files]
        return self._create_examples(texts)

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines):
        return [{'text': line} for line in lines]


class DbbookProcessor:
    def __init__(self, task_name):
        self.task_name = task_name

    def get_examples(self, mode='train'):
        df = pd.read_csv(
            PROCESSED_DATA_PATH / 'dbbook' / f"{mode}.tsv", sep='\t', header=None, 
            names=['user_id', 'item_id', 'label', 'user_genres', 'item_genre'])
        df["item_text"] = df.item_id \
            .map(lambda x: get_item_text(PROCESSED_DATA_PATH / "dbbook" / "texts" / f"{str(x)}.text"))
        df.fillna('', inplace=True)
        df = df.loc[df.item_text != '', :]
        
        return df.to_dict(orient='records')
    
    def get_structured_examples(self, mode='train'):
        df = pd.read_csv(PROCESSED_DATA_PATH / "dbbook" / f"{mode}.tsv", sep='\t', header=None, 
            names=['user_id', 'item_id', 'label', 'user_genres', 'item_genre'])
        
        item_features = pd.read_csv(
            INTERIM_DATA_PATH / "dbbook" / "item-prop" / "train.tsv", sep='\t', header=None,
            names=['item_id', 'item_author', 'item_genre', 'item_series', 'item_publisher', 'item_subject'])

        # merge df with item features
        df = df.merge(item_features, on='item_id', how='left')
        df.fillna('', inplace=True)

        df.rename(columns={'item_genre_x': 'item_genre'}, inplace=True)
        df.drop(['item_genre_y'], axis=1, inplace=True)

        # drop rows with empty features
        df = df.loc[df.item_author != '', :]
        df = df.loc[df.item_genre != '', :]
        df = df.loc[df.item_series != '', :]
        df = df.loc[df.item_publisher != '', :]
        df = df.loc[df.item_subject != '', :]

        return df.to_dict(orient='records')

    def get_labels(self):
        return [0, 1]
    
class DbbookRankingProcessor:
    def __init__(self, task_name):
        self.task_name = task_name

    def get_examples(self, user_id):
        df = pd.read_csv(
            PROCESSED_DATA_PATH / "dbbook" / "test.tsv", sep='\t', header=None, 
            names=['user_id', 'item_id', 'label', 'user_genres', 'item_genre'])

        df = df.loc[df.user_id == user_id]
        df.fillna('', inplace=True)
        df["item_text"] = df.item_id \
            .map(lambda x: get_item_text(PROCESSED_DATA_PATH / "dbbook" / "texts" / (str(x) + ".text")))
        
        df.drop(['label', 'item_genre'], axis=1, inplace=True)
        df = df.loc[df.item_text != '', :]
        
        return df.to_dict(orient='records')
    
    def get_structured_examples(self, user_id):
        df = pd.read_csv(PROCESSED_DATA_PATH / "dbbook" / f"test.tsv", sep='\t', header=None, 
            names=['user_id', 'item_id', 'label', 'user_genres', 'item_genre'])
        
        item_features = pd.read_csv(
            INTERIM_DATA_PATH / "dbbook" / "item-prop" / "train.tsv", sep='\t', header=None,
            names=['item_id', 'item_author', 'item_genre', 'item_series', 'item_publisher', 'item_subject'])

        df = df.loc[df.user_id == user_id]

        # merge df with item features
        df = df.merge(item_features, on='item_id', how='left')
        df.fillna('', inplace=True)

        df.rename(columns={'item_genre_x': 'item_genre'}, inplace=True)
        df.drop(['item_genre_y'], axis=1, inplace=True)

        # drop rows with empty features
        df = df.loc[df.item_author != '', :]
        df = df.loc[df.item_genre != '', :]
        df = df.loc[df.item_series != '', :]
        df = df.loc[df.item_publisher != '', :]
        df = df.loc[df.item_subject != '', :]

        df.drop(['label'], axis=1, inplace=True)

        return df.to_dict(orient='records')        

    def get_labels(self):
        return [0, 1]

processors_mapping = {
    "dbbook": DbbookProcessor("dbbook"),
    "dbbook_ranking": DbbookRankingProcessor("dbbook_ranking"),
    "truncate": TruncateProcessor("truncate")
}

num_labels_mapping = {
    "dbbook": 2,
    "dbbook_ranking": 2,
}

output_modes_mapping = {
    "dbbook": "classification",
    "dbbook_ranking": "classification",
}

def text_classification_metrics(_, preds, labels):
    return {"acc": (preds == labels).mean()}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "dbbook": text_classification_metrics,
    "dbbook_ranking": text_classification_metrics,
}
