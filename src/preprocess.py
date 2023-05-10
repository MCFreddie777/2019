import pandas as pd

from _helpers import constants
from _helpers import functions as hf
from _helpers.functions import verbose_print
from _helpers.preprocess import preprocess
from drop import main as drop


def main():
    """
    This function creates a parquet file which cleans up data
    """
    try:
        hf.require_files([constants.DROPPED_TRAIN, constants.DROPPED_TEST])
    except FileNotFoundError:
        drop()
    
    # Preprocess train
    verbose_print(f"Reading {constants.DROPPED_TRAIN}...")
    df_train = pd.read_parquet(constants.DROPPED_TRAIN)
    df_train = preprocess(df_train)
    df_train.to_parquet(constants.PREPROCESSED_TRAIN, index=False)
    verbose_print(f"Output saved to {constants.PREPROCESSED_TRAIN}.")
    
    # Preprocess test
    verbose_print(f"Reading {constants.DROPPED_TEST}...")
    df_test = pd.read_parquet(constants.DROPPED_TEST)
    df_test = preprocess(df_test)
    df_test.to_parquet(constants.PREPROCESSED_TEST, index=False)
    verbose_print(f"Output saved to {constants.PREPROCESSED_TEST}.")


if __name__ == "__main__":
    main()
