from threading import Thread

import numpy as np
import torch
from config import Config


class TestWorker(Thread):

    # TODO: maybe store (user_interacted_items, all_beer_ids, model) as static

    def __init__(self, hits, model, user_interacted_items, all_beer_ids, sequence):
        Thread.__init__(self)
        self.hits = hits
        self.model = model
        self.user_interacted_items = user_interacted_items
        self.all_beer_ids = all_beer_ids
        self.sequence = sequence

    def run(self):
        result = []
        # Get the work from the queue and expand the tuple
        for u, i in self.sequence:
            interacted_items = self.user_interacted_items[u]
            not_interacted_items = set(self.all_beer_ids) - set(interacted_items)
            selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
            test_items = selected_not_interacted + [i]

            predicted_labels = np.squeeze(self.model(torch.tensor([u] * 100),
                                                     torch.tensor(test_items)).detach().numpy())

            top_n_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:Config.hit_ratio].tolist()]

            if i in top_n_items:
                result.append(1)
            else:
                result.append(0)

        self.hits.extend(result)
