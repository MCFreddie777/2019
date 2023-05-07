import pandas as pd
import numpy as np
import random

from .._helpers import constants
from .._helpers import functions as f

def __random_gt(df):
    df_gt = df.copy()
    df_gt.loc[:, "reference"] = df_gt.loc[:, "impressions"].str.split("|").map(lambda x: random.choice(x))
    
    return df_gt

def __cheapest_gt(df):
    df_gt = df.copy()
    
    # Get cheapest price index
    target_impression_price_idx = df.loc[:, "prices"].str.split("|").apply(lambda x: np.argmin(list(map(int,x))))
    
    # Save impression at target_impression_price_idx
    df_gt.loc[:,'reference'] = [l[idx] for idx,l in zip(list(target_impression_price_idx), df_gt.loc[:,'impressions'].str.split("|"))]
    
    return df_gt
    
def main():
    """
    This function creates a ground truth file from a test file by choosing strategy of predicting real reference
    Original ground_truth.csv file from ACM Recsys 2019 Trivago Challenge is nowhere to be found
    """
    
    df_test = pd.read_csv(constants.TEST)
    df_target = f.get_target_rows(df_test)
    
    # df_gt = __random_gt(df_target)
    df_gt = __cheapest_gt(df_target)
    
    # Save ground truth to file
    df_gt.to_csv(constants.GROUND_TRUTH, index=False)
    
    
if __name__ == "__main__":
    main()
