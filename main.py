from preprocessing import train_test_split, process as process_data
from train import train
from test import test


def main():
    ratings = process_data()
    train_ratings, test_ratings = train_test_split(ratings)

    num_users = ratings['userId'].max()+1
    num_items = ratings['movieId'].max()+1
    all_movie_ids = ratings['movieId'].unique()

    model = train(num_users, num_items, train_ratings, all_movie_ids)
    test(ratings, test_ratings, model, all_movie_ids)


if __name__ == '__main__':
    main()
