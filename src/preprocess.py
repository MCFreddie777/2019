import pandas as pd

from _helpers import constants
from _helpers.preprocess import preprocess
from drop import main as drop


def main():
    """
    This function creates a parquet file which cleans up data
    """
    if not (constants.DROPPED_TRAIN.exists() or constants.DROPPED_TEST.exists()):
        drop()
    
    # Preprocess train
    print(f"Reading {constants.DROPPED_TRAIN}...")
    df_train = pd.read_parquet(constants.DROPPED_TRAIN)
    df_train = preprocess(df_train)
    df_train.to_parquet(constants.PREPROCESSED_TRAIN, index=False)
    print(f"Output saved to {constants.PREPROCESSED_TRAIN}.")
    
    # Preprocess test
    print(f"Reading {constants.DROPPED_TEST}...")
    df_test = pd.read_parquet(constants.DROPPED_TEST)
    df_test = preprocess(df_test)
    df_test.to_parquet(constants.PREPROCESSED_TEST, index=False)
    print(f"Output saved to {constants.PREPROCESSED_TEST}.")


if __name__ == "__main__":
    main()
