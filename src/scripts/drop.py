import pandas as pd

from .._helpers import constants
from .._helpers.drop import drop, __exclude_user_session_pairs

def main():
    """
    This function creates a parquet file which cleans up data
    """
    
    # Drop train dataset
    print(f"Reading {constants.TRAIN}...")
    df_train = pd.read_csv(constants.TRAIN)
    df_train, deleted_user_session_pairs = drop(df_train)
    df_train.to_parquet(constants.DROPPED_TRAIN, index=False)
    print(f"Output saved to {constants.DROPPED_TRAIN}.")
    
    # Remove dropped sessions from test dataset
    print(f"Reading {constants.TEST}...")
    df_test = pd.read_csv(constants.TEST)
    df_test = __exclude_user_session_pairs(df_test, deleted_user_session_pairs)
    print(f"Removed {len(deleted_user_session_pairs)} from test dataset.")
    df_test.to_parquet(constants.DROPPED_TEST, index=False)
    print(f"Output saved to {constants.DROPPED_TEST}.")
    
    # Remove dropped sessions from ground_truth dataset
    print(f"Reading {constants.GROUND_TRUTH}...")
    df_gt = pd.read_csv(constants.GROUND_TRUTH)
    df_gt = __exclude_user_session_pairs(df_gt, deleted_user_session_pairs)
    print(f"Removed {len(deleted_user_session_pairs)} from ground_truth.")
    df_gt.to_parquet(constants.DROPPED_GROUND_TRUTH, index=False)
    print(f"Output saved to {constants.DROPPED_GROUND_TRUTH}.")


if __name__ == "__main__":
    main()
