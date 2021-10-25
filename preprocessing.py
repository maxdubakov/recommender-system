import numpy as np
import pandas as pd

from config import *


def read():
    return pd.read_csv(data_path, parse_dates=['timestamp'])


def process():
    ratings = read()
    rand_user_ids = np.random.choice(ratings['userId'].unique(),
                                     size=int(len(ratings['userId'].unique()) * part_of_dataset_used),
                                     replace=False)
    ratings = ratings.loc[ratings['userId'].isin(rand_user_ids)]
    if verbose:
        print('There are {} rows of data from {} users'.format(len(ratings), len(rand_user_ids)))

    ratings['rank_latest'] = ratings \
        .groupby(['userId'])['timestamp'] \
        .rank(method='first', ascending=False)

    return ratings


def train_test_split(ratings):
    train_ratings = ratings[ratings['rank_latest'] != 1]
    test_ratings = ratings[ratings['rank_latest'] == 1]

    train_ratings = train_ratings[['userId', 'movieId', 'rating']]
    test_ratings = test_ratings[['userId', 'movieId', 'rating']]
    train_ratings.loc[:, 'rating'] = 1
    return train_ratings, test_ratings
