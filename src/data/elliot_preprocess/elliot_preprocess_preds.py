import os

import pandas as pd

from ..paths import ELLIOT_DATA_PATH, RESULTS_PATH

def main():
    results_files = ['flan-t5-base-dbbook-prompt-4.txt']

    for results_file in results_files:
        predictions = pd.read_csv(
            RESULTS_PATH / results_file, 
            sep='\t', header=None,
            names=['user_id', 'item_id'])
        
        test = pd.read_csv(
            ELLIOT_DATA_PATH / 'dbbook' / 'test.tsv', 
            sep='\t', header=None,
            names=['user_id', 'item_id', 'label'],
            dtype={'user_id': int, 'item_id': int, 'label': int})
        
        predictions_user_ids = set(predictions.user_id.unique())
        test_user_ids = set(test.user_id.unique())
        predictions_user_ids = predictions_user_ids - test_user_ids

        # drop from predictions such ids
        predictions = predictions.loc[~predictions.user_id.isin(predictions_user_ids), :]

        with open(RESULTS_PATH / f'elliot-{results_file}', "w") as f:
            for _, row in predictions.iterrows():
                f.write(f"{row['user_id']}\t{row['item_id']}\n")

if __name__ == '__main__':
    main()