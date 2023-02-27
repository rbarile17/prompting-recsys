import pandas as pd

from ..paths import RAW_DATA_PATH, INTERIM_DATA_PATH, ELLIOT_DATA_PATH

def get_item_text(item_path):
    try:
        with open(item_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ''

def main():
    df = pd.read_csv(
        RAW_DATA_PATH / 'dbbook' / 'test.tsv', 
        sep='\t', header=None,
        names=['user_id', 'item_id', 'label'])
    
    df = df.loc[df.label != 0, :]
    item_features = pd.read_csv(
        INTERIM_DATA_PATH / "dbbook" / "item-prop" / "train.tsv", sep='\t', header=None,
        names=['item_id', 'item_author', 'item_genre', 'item_series', 'item_publisher', 'item_subject'])

    # merge df with item features
    df = df.merge(item_features, on='item_id', how='left')
    df.fillna('', inplace=True)

    # drop rows with empty features
    df = df.loc[df.item_author != '', :]
    df = df.loc[df.item_genre != '', :]
    df = df.loc[df.item_series != '', :]
    df = df.loc[df.item_publisher != '', :]
    df = df.loc[df.item_subject != '', :]

    # drop all columns except user_id, item_id, label
    df.drop(columns=['item_author', 'item_genre', 'item_series', 'item_publisher', 'item_subject'], inplace=True)

    (ELLIOT_DATA_PATH / 'dbbook_structured').mkdir(parents=True, exist_ok=True)
    df.to_csv(ELLIOT_DATA_PATH / 'dbbook_structured' / 'test.tsv', sep='\t', header=False, index=False)

    df = pd.read_csv(
        RAW_DATA_PATH / 'dbbook' / 'train.tsv', 
        sep='\t', header=None,
        names=['user_id', 'item_id', 'label'])
    
    df = df.loc[df.label != 0, :]

    df.to_csv(ELLIOT_DATA_PATH / 'dbbook_structured' / 'train.tsv', sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()