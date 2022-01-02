from pathlib import Path

# data
DATA_DIR= (Path(__file__) / '../../../data').resolve()
TRAIN = DATA_DIR / 'train.csv'
TEST = DATA_DIR / 'test.csv'
DROPPED = DATA_DIR / 'dropped.parquet'
PREPROCESSED = DATA_DIR / 'preprocessed.parquet'
METADATA = DATA_DIR / 'item_metadata.csv'