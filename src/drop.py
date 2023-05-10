import pandas as pd

from _helpers import constants
from _helpers import functions as hf
from _helpers.functions import verbose_print
from _helpers.drop import drop, __exclude_user_session_pairs


def main():
    """
    This function creates a parquet file which cleans up data
    """
    hf.require_files([constants.TRAIN, constants.TEST])
    
    # Drop train dataset
    verbose_print(f"Reading {constants.TRAIN}...")
    df_train = pd.read_csv(constants.TRAIN)
    df_train, deleted_user_session_pairs = drop(df_train)
    df_train = hf.reduce_mem_usage(df_train)
    df_train.to_parquet(constants.DROPPED_TRAIN, index=False)
    verbose_print(f"Output saved to {constants.DROPPED_TRAIN}.")
    
    # Remove dropped sessions from test dataset
    verbose_print(f"Reading {constants.TEST}...")
    df_test = pd.read_csv(constants.TEST)
    df_test = __exclude_user_session_pairs(df_test, deleted_user_session_pairs)
    verbose_print(f"Removed {len(deleted_user_session_pairs)} from test dataset.")
    df_test = hf.reduce_mem_usage(df_test)
    df_test.to_parquet(constants.DROPPED_TEST, index=False)
    verbose_print(f"Output saved to {constants.DROPPED_TEST}.")


if __name__ == "__main__":
    main()
