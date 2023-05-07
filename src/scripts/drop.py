import pandas as pd

from .._helpers import constants
from .._helpers.drop import drop

def main():
    """
    This function creates a parquet file which cleans up data
    """
    
    df_train = pd.read_csv(constants.TRAIN)
    
    df_train = drop(df_train)
    
    df_train.to_parquet(constants.DROPPED, index=False)


if __name__ == "__main__":
    main()
