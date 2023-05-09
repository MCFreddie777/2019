from pathlib import Path
from dotenv import load_dotenv

from .functions import get_env

load_dotenv(override=True) # Load env variables from .env file

# Special
SUBSET=get_env('SUBSET', None)

# Directories
DATA_DIR= (Path(__file__) / '../../../data').resolve()
INPUT_DIR= DATA_DIR / 'input'
OUTPUT_DIR = DATA_DIR / 'output'
DROPPED_DIR = DATA_DIR / 'dropped'
PREPROCESSED_DIR = DATA_DIR / 'preprocessed'

# Files
TRAIN = INPUT_DIR / get_env('TRAIN', 'train.csv')
TEST = INPUT_DIR /  get_env('TEST', 'test.csv')
METADATA = INPUT_DIR / get_env('METADATA', 'item_metadata.csv')
GROUND_TRUTH = INPUT_DIR / get_env('GROUND_TRUTH', 'ground_truth.csv')

DROPPED_TRAIN = DROPPED_DIR / 'train.parquet'
DROPPED_TEST = DROPPED_DIR / 'test.parquet'

PREPROCESSED_TRAIN = PREPROCESSED_DIR / 'train.parquet'
PREPROCESSED_TEST = PREPROCESSED_DIR / 'test.parquet'

def DROPPED_SUBSET(subset_n,type):
    return DROPPED_DIR / f'{type}_subset_{subset_n}.parquet'

def PREPROCESSED_SUBSET(subset_n,type):
    return PREPROCESSED_DIR / f'{type}_subset_{subset_n}.parquet'

# Column names
GT_COLS = ["user_id", "session_id", "timestamp", "step"]
ITEM_REFERENCE_ACTION_TYPE_COLS = [
    'clickout item', 'interaction item deals', 'interaction item image',
    'interaction item info', 'interaction item rating', 'search for item'
]
