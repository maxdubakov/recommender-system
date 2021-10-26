import numpy as np
import torch

from config import hit_ratio, verbose


def test(ratings, test_ratings, model, all_beer_ids):
    if verbose:
        print('Preparing test data...')
    test_user_item_set = set(zip(test_ratings['user_id'], test_ratings['beer_id']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = ratings.groupby('user_id')['beer_id'].apply(list).to_dict()

    if verbose:
        print('Done.')
        print(f'Calculating The Hit Ratio @{hit_ratio}...')
    hits = []
    for (u, i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_beer_ids) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]

        predicted_labels = np.squeeze(model(torch.tensor([u]*100),
                                            torch.tensor(test_items)).detach().numpy())

        top_n_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:hit_ratio].tolist()]

        if i in top_n_items:
            hits.append(1)
        else:
            hits.append(0)

    if verbose:
        print('Done.')
    print("The Hit Ratio @{} is {:.2f}".format(hit_ratio, np.average(hits)))
