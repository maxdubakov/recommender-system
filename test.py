from queue import Queue
import time

import numpy as np
import torch

from config import Config
from threads.TestWorker import TestWorker
from utils.seq_operations import split_sequence


def test(ratings, test_ratings, model, all_beer_ids):
    if Config.verbose:
        print('Preparing test data...')
    test_user_item_set = set(zip(test_ratings['user_id'], test_ratings['beer_id']))
    sequences = list(split_sequence(list(test_user_item_set), Config.workers))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('user_id')['beer_id'].apply(list).to_dict()

    if Config.verbose:
        print('Done.')
        print(f'Calculating The Hit Ratio @{Config.hit_ratio}...')

    hits = []
    workers = []
    start_millis = round(time.time() * 1000)

    # Create 8 worker threads
    for i in range(Config.workers):
        worker = TestWorker(hits, model, user_interacted_items, all_beer_ids, sequences[i])
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        workers.append(worker)
        worker.start()

    for worker in workers:
        worker.join()

    required_time = round(time.time() * 1000) - start_millis
    if Config.verbose:
        print(f'Time required: {required_time}')
        print('Done.')

    hit_ratio = round(np.average(hits), 2)
    print(f'The Hit Ratio @{Config.hit_ratio} is {hit_ratio}')
    return hit_ratio
