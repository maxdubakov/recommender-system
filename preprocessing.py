import numpy as np
import pandas as pd

from config import Config


def read():
    return pd.read_csv(Config.data_path, parse_dates=Config.date_cols)


def str_to_id(ratings: pd.DataFrame, col_name: str, new_col_name):
    unique_values = ratings[col_name].unique()
    name_to_int = dict(zip(ratings[col_name].unique(), range(len(unique_values))))
    ratings[new_col_name] = ratings[col_name].apply(lambda name: name_to_int[name])
    return ratings


def process():
    if Config.verbose:
        print(f'Loading the dataset {Config.dataset_name}...')
    ratings = read()
    if Config.verbose:
        print('Done.')

    if Config.verbose:
        print(f'Processing the dataset {Config.dataset_name}...')
    ratings.drop('brewery_id', axis=1, inplace=True)
    ratings.drop('brewery_name', axis=1, inplace=True)
    ratings.drop('review_overall', axis=1, inplace=True)
    ratings.drop('review_aroma', axis=1, inplace=True)
    ratings.drop('review_appearance', axis=1, inplace=True)
    ratings.drop('beer_style', axis=1, inplace=True)
    ratings.drop('review_palate', axis=1, inplace=True)
    ratings.drop('beer_name', axis=1, inplace=True)
    ratings.drop('beer_abv', axis=1, inplace=True)
    ratings = str_to_id(ratings, 'review_profilename', 'user_id')
    ratings.drop('review_profilename', axis=1, inplace=True)

    rand_user_ids = np.random.choice(ratings['user_id'].unique(),
                                     size=int(len(ratings['user_id'].unique()) * Config.part_of_dataset_used),
                                     replace=False)
    ratings = ratings.loc[ratings['user_id'].isin(rand_user_ids)]
    if Config.verbose:
        print('There are {} rows of data from {} users'.format(len(ratings), len(rand_user_ids)))

    ratings[Config.rank_latest] = ratings \
        .groupby(['user_id'])['review_time'] \
        .rank(method='first', ascending=False)

    ratings.rename(columns={'beer_beerid': 'beer_id', 'review_taste': 'rating'}, inplace=True)
    if Config.verbose:
        print('Done.')
    return ratings


def train_test_split(ratings):
    if Config.verbose:
        print(f'Splitting the dataset {Config.dataset_name}...')
    train_ratings = ratings[ratings[Config.rank_latest] != 1]
    test_ratings = ratings[ratings[Config.rank_latest] == 1]

    train_ratings = train_ratings[['user_id', 'beer_id', 'rating']]
    test_ratings = test_ratings[['user_id', 'beer_id', 'rating']]
    train_ratings.loc[:, 'rating'] = 1
    if Config.verbose:
        print('Done.')
    return train_ratings, test_ratings
