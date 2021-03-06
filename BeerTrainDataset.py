import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config


class BeerTrainDataset(Dataset):

    GENERATED = False
    USERS, ITEMS, LABELS = [], [], []
    """
    Beer PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe containing the beer ratings
        all_beer_ids (list): List containing all beer ids
    """
    def __init__(self, ratings, all_beer_ids):
        self.users, self.items, self.labels = self.get_dataset(ratings, all_beer_ids)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings, all_beer_ids):
        if BeerTrainDataset.GENERATED:
            return self.get_generated()
        else:
            return self.generate_dataset(ratings, all_beer_ids)

    def generate_dataset(self, ratings, all_beer_ids):
        if Config.verbose:
            print('Appending negative example to use implicit feedback...')
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['user_id'], ratings['beer_id']))

        num_negatives = 4
        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(num_negatives):
                negative_item = np.random.choice(all_beer_ids)
                while (u, negative_item) in user_item_set:
                    negative_item = np.random.choice(all_beer_ids)
                users.append(u)
                items.append(negative_item)
                labels.append(0)

        if Config.verbose:
            print('Done.')

        BeerTrainDataset.USERS, BeerTrainDataset.ITEMS, BeerTrainDataset.LABELS = \
            torch.tensor(users), torch.tensor(items), torch.tensor(labels)
        BeerTrainDataset.GENERATED = True

        return BeerTrainDataset.USERS, BeerTrainDataset.ITEMS, BeerTrainDataset.LABELS

    def get_generated(self):
        return BeerTrainDataset.USERS, BeerTrainDataset.ITEMS, BeerTrainDataset.LABELS
