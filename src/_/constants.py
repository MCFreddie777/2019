from pathlib import Path

# data
DATA_DIR= (Path(__file__) / '../../../data').resolve()
TRAIN = DATA_DIR / 'train.csv'
TEST = DATA_DIR / 'test.csv'
TRAIN_DROPPED = DATA_DIR / 'train-dropped.parquet'
TEST_DROPPED = DATA_DIR / 'test-dropped.parquet'
TRAIN_PREPROCESSED = DATA_DIR / 'train-preprocessed.parquet'
TEST_PREPROCESSED = DATA_DIR / 'test-preprocessed.parquet'
SCHEMA = DATA_DIR / 'schema.json'