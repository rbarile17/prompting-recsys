import pandas as pd

from .paths import INTERIM_DATA_PATH, PROCESSED_DATA_PATH

def main():
    books_genres = pd.read_csv(
        INTERIM_DATA_PATH / "dbbook" / "item-prop" / "train.tsv", 
        sep="\t", header=None, names=['item_id', 'author', 'genre', 'series', 'publisher', 'subject'],
        dtype={'item_id': 'int64', 'author': 'str', 'genre': 'str', 'series': 'str', 'publisher': 'str', 'subject': 'str'})
    books_genres = books_genres.drop(columns=["author", "series", "publisher", "subject"])

    books_ratings = pd.read_csv(INTERIM_DATA_PATH / "dbbook" / "train.tsv", sep="\t", header=None)
    books_ratings.columns = ["user_id", "item_id", "label"]
    
    positive_books_ratings = books_ratings.loc[books_ratings["label"] == 1]
    positive_books_ratings = pd.merge(positive_books_ratings, books_genres, on="item_id", how="left")
    positive_books_ratings.fillna("", inplace=True)

    liked_genres_by_user = positive_books_ratings.groupby("user_id")["genre"] \
        .apply(list).reset_index()
    liked_genres_by_user.columns = ["user_id", "liked_genres"]

    liked_genres_by_user["liked_genres"] = liked_genres_by_user["liked_genres"] \
        .map(lambda x: [str(element) for element in x if element != ""])
    
    for split_name in ["train", "dev", "test"]:
        split = pd.read_csv(INTERIM_DATA_PATH / "dbbook" / f"{split_name}.tsv", sep="\t", header=None)
        split.columns = ["user_id", "item_id", "label"]

        split = split.merge(liked_genres_by_user, on="user_id", how="left")
        split.fillna("", inplace=True)

        split = pd.merge(split, books_genres, on="item_id", how="left")
        split.fillna("", inplace=True)

        split["liked_genres"] = split["liked_genres"].map(lambda x: list(set(x)))
        split["liked_genres"] = split["liked_genres"].map(lambda x: (", ".join(x)))

        split = split.sort_values(by=["user_id", "item_id"])

        (PROCESSED_DATA_PATH / "dbbook").mkdir(parents=True, exist_ok=True)
        split.to_csv(PROCESSED_DATA_PATH / "dbbook" / f"{split_name}.tsv", sep="\t", header=None, index=False)


if __name__ == '__main__':
    main()