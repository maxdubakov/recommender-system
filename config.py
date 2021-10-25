"""This is a config file which possesses all changeable variables"""

data_path = './data/rating.csv'
part_of_dataset_used = 0.3
verbose = True
num_negatives = 4
batch_size = 512
save_model = True
epochs = 5
gpus = 0
reload_dataloaders_every_epoch = True
progress_bar_refresh_rate = 50
checkpoint_callback = False
