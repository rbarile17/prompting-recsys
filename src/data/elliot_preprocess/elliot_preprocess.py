import pandas as pd

from ..paths import RAW_DATA_PATH, ELLIOT_DATA_PATH

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
    df["item_text"] = df.item_id \
        .map(lambda x: get_item_text(RAW_DATA_PATH / "dbbook" / "texts" / f"{str(x)}.text"))
    
    df.fillna('', inplace=True)
    df = df.loc[df.item_text != '', :]

    # drop item_text column
    df.drop(columns=['item_text'], inplace=True)

    (ELLIOT_DATA_PATH / 'dbbook').mkdir(parents=True, exist_ok=True)
    df.to_csv(ELLIOT_DATA_PATH / 'dbbook' / 'test.tsv', sep='\t', header=False, index=False)

    df = pd.read_csv(
        RAW_DATA_PATH / 'dbbook' / 'train.tsv', 
        sep='\t', header=None,
        names=['user_id', 'item_id', 'label'])
    
    df = df.loc[df.label != 0, :]

    (ELLIOT_DATA_PATH / 'dbbook').mkdir(parents=True, exist_ok=True)
    df.to_csv(ELLIOT_DATA_PATH / 'dbbook' / 'train.tsv', sep='\t', header=False, index=False)


if __name__ == '__main__':
    main()