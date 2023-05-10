import os
import glob
import pandas as pd
import numpy as np

from . import constants


def get_env(name, default_value, required=False):
    value = os.getenv(name, default_value)
    
    # Sanitize the input (remove leading/trailing whitespaces)
    if isinstance(value, str):
        value = value.strip() if value else None
    
    if (value is None):
        if (default_value is None and required is True):
            raise EnvironmentError(
                f'Required environment variable "{name}" is not set.'
            )
        else:
            value = default_value
    
    # Convert string input to integer if possible
    try:
        value = int(value)
    except (TypeError, ValueError):
        pass
    
    return value


def get_target_rows(df):
    """
    Target rows which need to be predicted
    """
    mask = (df["action_type"] == "clickout item") & df["reference"].isnull()
    df_target = df[mask]
    
    return df_target


def explode(df_in, columns):
    """Separate string in each column in columns and explode elements into multiple rows."""
    
    df = df_in.copy()
    
    for col in columns:
        df.loc[:, col] = df.loc[:, col].str.split("|")
    
    df2 = pd.DataFrame(
        {col: np.repeat(df[col].to_numpy(),
                        df[columns[0]].str.len())
         for col in df.columns.drop(columns)}
    )
    
    for col in columns:
        df2.loc[:, col] = np.concatenate(df.loc[:, col].to_numpy())
    
    return df2


def cast(df, column, type):
    df2 = df.copy()
    df2.loc[:, column] = df2[column].astype(type)
    return df2


def reorder_column(df, from_idx, to_idx):
    """
    Function which reorders columns in Pandas DataFrame
    """
    columns = df.columns.tolist()
    column_to_move = columns.pop(from_idx)
    columns.insert(to_idx, column_to_move)
    
    return df[columns]


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def verbose_print(str, verbose=get_env('VERBOSE', False)):
    if verbose:
        print(str)


def require_files(files):
    """
    Checks whether the given file exists
    """
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)


def load_preprocessed_dataset(type):
    pattern = str(constants.PREPROCESSED_DIR) + f'/{type}_*.parquet'
    files = glob.glob(pattern)
    
    if not (len(files)):
        raise FileNotFoundError(pattern)
    
    dfs = []
    
    for file in files:
        df = pd.read_parquet(file)
        dfs.append(df)
    
    return pd.concat(dfs)
