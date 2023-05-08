import pandas as pd

from .._helpers import constants
from .._helpers import functions as hf
from .drop import main as drop

def __subset_users(df, subset):
    """
    Return a new df window, which excludes the overlap of subset users pairs
    """
    mask = df.set_index(['user_id']).index.isin(subset.set_index(['user_id']).index)
    return df[mask]

def main():
    """
    This function creates a subset from dataset used just for testing purposes
    We need to do it in reverse order - first subset test dataset, then train.
    (to be sure users in picked subset are present in both test and train)
    """
    
    # Number of unique user-session pairs (resulting dataframe may contain larger number of rows)
    n = hf.get_env('SUBSET', 200, required=True)
    
    if not (constants.DROPPED_TRAIN.exists() or constants.DROPPED_TEST.exists()):
        drop()
    
    # Subset test dataset
    print(f"Reading {constants.DROPPED_TEST}...")
    df_test = pd.read_parquet(constants.DROPPED_TEST)
    
    subset_user_session_pairs = df_test[['user_id', 'session_id']].drop_duplicates().head(n)
    print(f"Picked {len(subset_user_session_pairs)} from test dataset.")
    
    subset_test = __subset_users(df_test, subset_user_session_pairs)
    subset_test.to_parquet(constants.DROPPED_SUBSET(n, 'test'), index=False)
    print(f"Output saved to {constants.DROPPED_SUBSET(n, 'test')}.")
    
    # Subset train dataset
    print(f"Reading {constants.DROPPED_TRAIN}...")
    df_train = pd.read_parquet(constants.DROPPED_TRAIN)
    
    subset_train = __subset_users(df_train, subset_user_session_pairs)
    subset_train.to_parquet(constants.DROPPED_SUBSET(n, 'train'), index=False)
    print(f"Output saved to {constants.DROPPED_SUBSET(n, 'train')}.")
    
    # Subset ground_truth
    print(f"Reading {constants.DROPPED_GROUND_TRUTH}...")
    df_gt = pd.read_parquet(constants.DROPPED_GROUND_TRUTH)
    
    subset_gt = __subset_users(df_gt, subset_user_session_pairs)
    subset_gt.to_parquet(constants.DROPPED_SUBSET(n, 'ground_truth'), index=False)
    print(f"Output saved to {constants.DROPPED_SUBSET(n, 'ground_truth')}.")

if __name__ == "__main__":
    main()
