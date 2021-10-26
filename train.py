from NCF import NCF
import pytorch_lightning as pl
import pickle as pkl

from config import *


def train(num_users, num_items, train_ratings, all_beer_ids):
    model = NCF(num_users, num_items, train_ratings, all_beer_ids)

    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus,
                         reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
                         progress_bar_refresh_rate=progress_bar_refresh_rate,
                         logger=verbose, checkpoint_callback=checkpoint_callback)
    if verbose:
        print('Training the model...')

    trainer.fit(model)

    if save_model:
        with open(save_path, 'wb+') as f:
            pkl.dump(model, f, protocol=pkl.HIGHEST_PROTOCOL)
        if verbose:
            print(f'The NFC model has been saved to the {save_path}')

    return model
