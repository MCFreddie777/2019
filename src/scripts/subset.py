import pandas as pd
from dotenv import load_dotenv

from .._helpers import constants
from .._helpers import functions as hf
from .._helpers.drop import drop

def main():
    """
    This function creates a subset from train dataframe used for testing
    """
    
    # Load env variables from .env file
    load_dotenv()
    
    if not constants.DROPPED.exists():
        df_train = pd.read_csv(constants.TRAIN)
        df_train = drop(df_train)
        df_train.to_parquet(constants.DROPPED, index=False)
    else:
        df_train = pd.read_parquet(constants.DROPPED)

    # Number of unique user / session pairs (resulting dataframe may contain larger number of rows)
    n = hf.get_env('SUBSET', 200)
    
    subset_user_session_pairs = df_train[['user_id', 'session_id']].drop_duplicates().head(n)
    subset = pd.merge(df_train, subset_user_session_pairs, on=['user_id', 'session_id'])
    
    subset.to_parquet(constants.DROPPED_SUBSET(n), index=False)


if __name__ == "__main__":
    main()
