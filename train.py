from NCF import NCF
import pytorch_lightning as pl
import pickle as pkl

from config import *


def train(num_users, num_items, train_ratings, all_movie_ids):
    model = NCF(num_users, num_items, train_ratings, all_movie_ids)

    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus,
                         reload_dataloaders_every_epoch=reload_dataloaders_every_epoch,
                         progress_bar_refresh_rate=progress_bar_refresh_rate,
                         logger=verbose, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    if save_model:
        with open('./model/model.pkl', 'wb+') as f:
            pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)

    return model
