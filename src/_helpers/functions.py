import os
import pandas as pd
import numpy as np


def get_env(name, default_value, required=False):
    value = os.getenv(name, default_value)
    
    # Sanitize the input (remove leading/trailing whitespaces)
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
