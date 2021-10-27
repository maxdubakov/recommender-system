from threading import Thread

import numpy as np
import torch
from config import Config


class TestWorker(Thread):

    # TODO: maybe store (user_interacted_items, all_beer_ids, model) as static

    def __init__(self, queue, user_interacted_items, all_beer_ids, model):
        Thread.__init__(self)
        self.queue = queue
        self.user_interacted_items = user_interacted_items
        self.all_beer_ids = all_beer_ids
        self.model = model

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            u, i, hits = self.queue.get()
            try:
                interacted_items = self.user_interacted_items[u]
                not_interacted_items = set(self.all_beer_ids) - set(interacted_items)
                selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
                test_items = selected_not_interacted + [i]

                predicted_labels = np.squeeze(self.model(torch.tensor([u] * 100),
                                                         torch.tensor(test_items)).detach().numpy())

                top_n_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:Config.hit_ratio].tolist()]

                if i in top_n_items:
                    hits.append(1)
                else:
                    hits.append(0)
            finally:
                self.queue.task_done()
