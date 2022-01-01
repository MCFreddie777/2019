import pandas as pd
from functools import partial

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


def explode_multivalue_attributes(df,columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: x.split('|') if x != None else x)
        
    return df

def encode_categorical(df,columns):
    for col in columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    
    return df

def one_hot_encode(df,columns):
    return pd.get_dummies(data=df, columns=columns)
    
    
def remove_step_column(df):
    df.sort_values(['session_id','step'])
    # no longer needed column
    del df['step']

    return df

def preprocess(path,backup_path):    

    #check if data exists
    if not path.exists():
        df = drop(backup_path);
        df.to_parquet(path, index=False)
    else:
        df = pd.read_parquet(path)

    
    functions = [
        partial(explode_multivalue_attributes,columns=['impressions','prices','current_filters']),
        partial(encode_categorical, columns=['user_id','session_id', 'action_type', 'platform', 'city']),
        partial(one_hot_encode, columns=['device']),
        remove_step_column
    ];
    
    for fun in functions:
        df = fun(df);
    
    return df