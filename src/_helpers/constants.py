from pathlib import Path

# Directories
DATA_DIR= (Path(__file__) / '../../../data').resolve()
RAW_DIR= DATA_DIR / 'raw'
OUTPUT_DIR = DATA_DIR / 'output'

# Files
TRAIN = RAW_DIR / 'train.csv'
TEST = RAW_DIR / 'test.csv'
METADATA = RAW_DIR / 'item_metadata.csv'
