import pandas as pd
from config import Config

df = pd.read_csv(Config.data_path, parse_dates=Config.date_cols)

print(len((list(df['beer_name'].unique()))))
