"""This is a config file which possesses all changeable variables"""

data_path = './data/beer_reviews.csv'
dataset_name = 'Beer'
part_of_dataset_used = 1
verbose = True
num_negatives = 4
hit_ratio = 10
batch_size = 512
save_model = True
epochs = 5
gpus = 0
reload_dataloaders_every_n_epochs = 1
progress_bar_refresh_rate = 50
checkpoint_callback = False
date_cols = ['review_time']
rank_latest = 'rank_latest'
model_id = 2
save_path = f'./models/model_{model_id}.pkl'
