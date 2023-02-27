import pandas as pd
import yaml

from .paths import RAW_DATA_PATH, INTERIM_DATA_PATH

def main(params):
    dataset = params["dataset"]
    train = pd.read_csv(RAW_DATA_PATH / dataset / "train.tsv", sep="\t", header=None)
    train.columns = ["user_id", "item_id", "label"]
    
    dev_sets = []
    for _, group in train.groupby("user_id"):
        dev_group = group.sample(frac=params["dev_size"], random_state=params["seed"])
        dev_sets.append(dev_group)
        train = train.drop(dev_group.index)

    dev = pd.concat(dev_sets)

    (INTERIM_DATA_PATH / "dbbook").mkdir(parents=True, exist_ok=True)
    train.to_csv(INTERIM_DATA_PATH / dataset / "train.tsv", sep="\t", index=False, header=False)
    dev.to_csv(INTERIM_DATA_PATH / dataset / "dev.tsv", sep="\t", index=False, header=False)

    test = pd.read_csv(RAW_DATA_PATH / dataset / "test.tsv", sep="\t", header=None)
    test.to_csv(INTERIM_DATA_PATH / dataset / "test.tsv", sep="\t", index=False, header=False)
    
if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train_dev_split"]
    main(params)