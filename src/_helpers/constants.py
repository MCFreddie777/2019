from pathlib import Path
from dotenv import load_dotenv

from .functions import get_env

load_dotenv(override=True)  # Load env variables from .env file

# Directories
DATA_DIR = (Path(__file__) / '../../../data').resolve()
INPUT_DIR = DATA_DIR / 'input'
OUTPUT_DIR = DATA_DIR / 'output'
DROPPED_DIR = DATA_DIR / 'dropped'
PREPROCESSED_DIR = DATA_DIR / 'preprocessed'

# Files
TRAIN = INPUT_DIR / get_env('TRAIN', 'train.csv')
TEST = INPUT_DIR / get_env('TEST', 'test.csv')
METADATA = INPUT_DIR / get_env('METADATA', 'item_metadata.csv')
GROUND_TRUTH = INPUT_DIR / get_env('GROUND_TRUTH', 'test_ground_truth.csv')

DROPPED_TRAIN = DROPPED_DIR / 'train.parquet'


def PREPROCESSED(n, type):
    return PREPROCESSED_DIR / f'{type}_{n}.parquet'


# Column names
ITEM_REFERENCE_ACTION_TYPE_COLS = [
    'clickout item', 'interaction item deals', 'interaction item image',
    'interaction item info', 'interaction item rating', 'search for item'
]
