import re
import os
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm

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

def drop_single_action_users(df):
    user_id_count = df.user_id.value_counts().reset_index(name="count").rename(columns={'index': 'user_id'})
    df = df[~df['user_id'].isin(user_id_count[(user_id_count['count'] == 1)]['user_id'])]
    
    return df

def drop_single_and_excessive_step_sessions(df,remove_single,upper_limit):  
    session_action_count = df.groupby(['session_id'])['action_type'].count().reset_index(name="count")
    df = df[~df['session_id'].isin(session_action_count[(session_action_count['count'] == (1 if remove_single else 0)) | (session_action_count['count'] > upper_limit)]['session_id'])]
    
    return df

def drop_duplicate_steps_in_session(df):
    df = df.reset_index().drop_duplicates(subset=['session_id','step'],keep='last').set_index('index')
    
    return df
    

def drop(path):
    
    functions = [ 
        drop_single_action_users,
        partial(drop_single_and_excessive_step_sessions, remove_single=True, upper_limit=100),
        drop_duplicate_steps_in_session
    ];
    
    df = pd.read_csv(path,sep=",")
    origin_len = len(df)
    
    for fun in functions:
        prev_len = len(df);
        df = fun(df);
        cur_len = len(df);
        print(f"{fun.__name__ if hasattr(fun,'__name__') else fun.func.__name__}: Dropped {prev_len - cur_len} records.")
    
    print(f"{path.name} - Previously {origin_len}, now {cur_len} (Dropped {origin_len - cur_len} in total).");
    return df


def explode_multivalue_attributes(df,columns,cast=None):
    for col in columns:
        if cast is None:
            df[col] = df[col].apply(lambda x: x.split('|') if x != None else x)
        else:
            df[col] = df[col].apply(lambda x: list(map(cast,x.split('|'))) if x != None else x)
        
    return df

def encode_categorical(df,columns):
    for col in columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    
    return df

def one_hot_encode(df,columns):
    return pd.get_dummies(data=df, columns=columns)

def impression_index(df):
    def _get_imp_index(reference,impressions):
        if impressions is not None:
            return next((index for index,impr in enumerate(impressions) if impr == reference),np.nan)
        return np.nan
    
    df['impression_index'] = df.apply(lambda x: _get_imp_index(x['reference'], x['impressions']), axis=1)

    return df
    
    
def calculate_price(df):
    df['price'] = df.apply(lambda x: x['prices'][int(x['impression_index'])] if ~np.isnan(x['impression_index']) else np.nan, axis=1)
    df['mean_price'] = df['prices'].apply(lambda x: np.mean(x) if x is not None else np.nan)
    
    return df

def parse_stars(string):
    result = re.search(r'^(\d) star$',string, flags=re.IGNORECASE)
    if result != None:
        return int(result.group(1))
    
def parse_ratings(df):
    rating_map = {
    'Satisfactory Rating': 7.0,
    'Good Rating': 7.5,
    'Very Good Rating': 8.0,
    'Excellent Rating': 8.5
    }
    
    for rating_key in rating_map:
        df.loc[df['properties'].apply(lambda x: rating_key in x),'rating'] = rating_map[rating_key]    
        
    return df


def merge_meta(df,meta):
    for index,_ in enumerate(tqdm(df.itertuples(), desc="Iterating rows", total=df.shape[0])):
        row = df.iloc[int(index)]
        try:
            item = meta.loc[int(row['reference'])]
            row['stars'] = item['stars']
            row['rating'] = item['rating']
        except:
            row['stars'] = np.nan
            row['rating'] = np.nan
             
    return df
    
def preprocess_meta(df,meta):
    meta.set_index('item_id',inplace=True)
    meta['properties'] = meta['properties'].apply(lambda x: x.split('|'))
    meta['stars'] = meta['properties'].apply(lambda props: next((i for i in list(map(parse_stars, props)) if i is not None), np.nan))
    meta = parse_ratings(meta)
    df = merge_meta(df,meta)
    
    return df    

def preprocess(path,backup_path,meta_path):    

    #check if data exists
    if not path.exists():
        df = drop(backup_path);
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)
        
    meta = pd.read_csv(meta_path)
    
    functions = [
        partial(explode_multivalue_attributes,columns=['impressions','current_filters']),
        partial(explode_multivalue_attributes,columns=['prices'], cast=int),
        impression_index,
        calculate_price,
        partial(encode_categorical, columns=['session_id', 'action_type', 'platform', 'city']),
        partial(one_hot_encode, columns=['device']),
        partial(preprocess_meta,meta=meta),
    ];
    
    for fun in functions:
        print(f"Running {fun.__name__ if hasattr(fun,'__name__') else fun.func.__name__}...")
        df = fun(df);
    
    return df

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
