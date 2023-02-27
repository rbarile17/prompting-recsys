import pandas as pd

from .paths import RAW_DATA_PATH, INTERIM_DATA_PATH

prop = pd.read_csv(
    RAW_DATA_PATH / 'dbbook' / 'mapping_entities.tsv', 
    sep='\t', header=1, names=['id', 'prop'], 
    dtype={'id': 'int64', 'prop': 'str'})
map = pd.read_csv(
    RAW_DATA_PATH / 'dbbook' / 'item-prop/train.tsv', 
    sep='\t', header=None, names=['item_id', 'id', 'type'], 
    dtype={'item_id': 'int64', 'id': 'int64', 'type': 'int64'})

item = map[['item_id']].copy()
item = item.drop_duplicates()

map_prop = pd.merge(prop, map, left_on='id', right_on='id', how='inner')
map_prop = map_prop.drop(columns=['id'])

author = map_prop.loc[map_prop['type'] == 2]
def add_author(row):
    prop = author.loc[author['item_id'] == row['item_id']]
    if prop.empty:
        return None
    prop['prop'] = prop['prop'].str.split('http://dbpedia.org/resource/').str[1]
    prop = prop['prop'].values[0]
    return prop

item.insert(1, 'author', None)
item['author'] = item.apply(add_author, axis=1)

genre = map_prop.loc[map_prop['type'] == 4]
def add_genre(row):
    prop = genre.loc[genre['item_id'] == row['item_id']]
    if prop.empty:
        return None
    prop['prop'] = prop['prop'].str.split('http://dbpedia.org/resource/').str[1]
    prop['prop'] = prop['prop'].str.replace('_(genre)', '', regex=False)

    prop = prop['prop'].values[0]
    return prop

item.insert(2, 'genre', None)
item['genre'] = item.apply(add_genre, axis=1)

series = map_prop.loc[map_prop['type'] == 3]
def add_series(row):
    prop = series.loc[series['item_id'] == row['item_id']]
    if prop.empty:
        return None
    prop['prop'] = prop['prop'].str.split('http://dbpedia.org/resource/').str[1]
    prop = prop['prop'].values[0]
    return prop

item.insert(3, 'series', None)
item['series'] = item.apply(add_series, axis=1)

publisher = map_prop.loc[map_prop['type'] == 5]
def add_publisher(row):
    prop = publisher.loc[publisher['item_id'] == row['item_id']]
    if prop.empty:
        return None
    prop['prop'] = prop['prop'].str.split('http://dbpedia.org/resource/').str[1]
    prop = prop['prop'].values[0]
    return prop

item.insert(4, 'publisher', None)
item['publisher'] = item.apply(add_publisher, axis=1)

subject = map_prop.loc[map_prop['type'] == 7]
def add_subject(row):
    prop = subject.loc[subject['item_id'] == row['item_id']]
    if prop.empty:
        return None
    result = ''
    for row in prop.itertuples():
        sub = row[1].split('http://dbpedia.org/resource/')[1]
        sub = sub.split('Category:')[1]
        result += sub + ','
    return result

item.insert(5, 'subject', None)
item['subject'] = item.apply(add_subject, axis=1)

(INTERIM_DATA_PATH / 'dbbook' / 'item-prop').mkdir(parents=True, exist_ok=True)
item.to_csv(INTERIM_DATA_PATH / 'dbbook' / 'item-prop' / 'train.tsv', sep='\t', index=False, header=False)