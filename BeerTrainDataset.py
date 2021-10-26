import numpy as np
import torch
from torch.utils.data import Dataset

from config import verbose

class BeerTrainDataset(Dataset):
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
        if verbose:
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

        if verbose:
            print('Done.')
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
