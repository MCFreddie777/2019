import pandas as pd

from . import functions as f
from _helpers import constants


def main(df_subm):
    """
    Function for verifying if submitted file is in correct format ready for scoring
    """
    df_test = pd.read_csv(constants.TEST)
    
    print('Checking for required columns in the submission file...')
    check_cols = f.check_columns(df_subm)
    f.check_passed(check_cols)
    
    print('Checking for duplicate sessions in the submission file...')
    check_dupl = f.check_duplicates(df_subm)
    f.check_passed(check_dupl)
    
    print('Checking that all the required sessions are present in submission...')
    check_sess = f.check_sessions(df_subm, df_test)
    f.check_passed(check_sess)
    
    if all([check_cols, check_dupl, check_sess]):
        print('All checks passed')
    else:
        raise Exception('One or more checks failed')
