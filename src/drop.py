import pandas as pd

from _helpers import constants
from _helpers import functions as hf
from _helpers.functions import verbose_print
from _helpers.drop import drop


def main():
    """
    This function creates a parquet file which cleans up data
    """
    
    verbose_print(f"Reading {constants.TRAIN}...")
    df_train = pd.read_csv(constants.TRAIN)
    df_train, deleted_user_session_pairs = drop(df_train)
    df_train = hf.reduce_mem_usage(df_train)
    df_train.to_parquet(constants.DROPPED_TRAIN, index=False)
    verbose_print(f"Output saved to {constants.DROPPED_TRAIN}.")


if __name__ == "__main__":
    main()
