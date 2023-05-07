from pathlib import Path

# Directories
DATA_DIR= (Path(__file__) / '../../../data').resolve()
RAW_DIR= DATA_DIR / 'raw'
OUTPUT_DIR = DATA_DIR / 'output'
DROPPED_DIR = DATA_DIR / 'dropped'

# Files
TRAIN = RAW_DIR / 'train.csv'
TEST = RAW_DIR / 'test.csv'
METADATA = RAW_DIR / 'item_metadata.csv'
GROUND_TRUTH = RAW_DIR / 'ground_truth.csv'
DROPPED = DROPPED_DIR / 'train.parquet'
def DROPPED_SUBSET(subset_n):
    return DROPPED.with_name(f'train_subset_{subset_n}.parquet')
