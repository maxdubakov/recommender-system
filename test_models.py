import time
import pandas as pd

from main import main
from config import Config

EMBEDDING_DIMS = [8, 16]
EPOCHS = [4, 8, 16, 24]
NUM_NEGATIVES = [2, 4, 6]
RESULT_PATH = './data/results.csv'


def test_models():
    results = []
    Config.verbose = False
    Config.save_model = False
    print('EMBEDDING_DIMS\tEPOCHS\tNUM_NEGATIVES\tBATCH_SIZES\tRESULT_PATH\tHIT_RATIO\tEXEC_TIME_MINS')
    for _embedding_dim in EMBEDDING_DIMS:
        for _epochs in EPOCHS:
            for _num_negatives in NUM_NEGATIVES:
                try:
                    Config(_num_negatives, 512, _embedding_dim, _epochs, _embedding_dim*2)
                    start_secs = round(time.time())
                    hit_ratio = main()
                    exec_time_mins = round((round(time.time()) - start_secs) / 60, 2)
                    current_result = [Config.embedding_dim, Config.epochs, Config.num_negatives, Config.batch_size,
                                      hit_ratio, exec_time_mins]
                    results.append(current_result)
                    print(current_result)
                except Exception as e:
                    print(e)
                    print('Impossible combination:')
                    print([Config.embedding_dim, Config.epochs, Config.num_negatives, Config.batch_size])
    pd.DataFrame(results,
                 columns=['EMBEDDING_DIMS', 'EPOCHS', 'NUM_NEGATIVES', 'BATCH_SIZES', 'HIT_RATIO',
                          'EXEC_TIME_MINS']).to_csv(RESULT_PATH, index=False)


if __name__ == '__main__':
    test_models()
