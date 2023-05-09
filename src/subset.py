import pandas as pd

from _helpers import constants
from _helpers import functions as hf
from _helpers.preprocess import preprocess
from drop import main as drop

def __subset_users(df, subset):
    """
    Return a new df window, which excludes the overlap of subset users pairs
    """
    mask = df.set_index(['user_id']).index.isin(subset.set_index(['user_id']).index)
    return df[mask]

def main(preprocess_data=False):
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
    
    if (preprocess_data is True):
        # Preprocess train
        print(f"Preprocessing train...")
        subset_preprocessed_train = preprocess(subset_train)
        subset_preprocessed_train.to_parquet(constants.PREPROCESSED_SUBSET(n, 'train'), index=False)
        print(f"Preprocessed output saved to {constants.PREPROCESSED_SUBSET(n, 'train')}.")
        
        # Preprocess test
        print(f"Preprocessing test...")
        subset_preprocessed_test = preprocess(subset_test)
        subset_preprocessed_test.to_parquet(constants.PREPROCESSED_SUBSET(n, 'test'), index=False)
        print(f"Preprocessed output saved to {constants.PREPROCESSED_SUBSET(n, 'test')}.")
        

if __name__ == "__main__":
    main(preprocess_data=True)
