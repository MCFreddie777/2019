import os
import tempfile
import pandas as pd
import pyarrow as pa
import apache_beam as beam
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.statistics import stats_options as options
from tensorflow_data_validation.utils import stats_util
from tfx_bsl.arrow.array_util import ToSingletonListArray
from functools import partial
from google.protobuf import json_format
import json
from tensorflow.python.lib.io import file_io

def drop_single_action_users(df):
    user_id_count = df.user_id.value_counts().reset_index(name="count").rename(columns={'index': 'user_id'})
    df = df[~df['user_id'].isin(user_id_count[(user_id_count['count'] == 1)]['user_id'])]
    
    return df

def drop_single_and_excessive_step_sessions(df):
    SINGLE = 1;
    UPPER_LIMIT = 100;
   
    session_action_count = df.groupby(['session_id'])['action_type'].count().reset_index(name="count")
    df = df[~df['session_id'].isin(session_action_count[(session_action_count['count'] == SINGLE) | (session_action_count['count'] > UPPER_LIMIT)]['session_id'])]
    
    return df

def drop_duplicate_steps_in_session(df):
    df = df.reset_index().drop_duplicates(subset=['session_id','step'],keep='last').set_index('index')
    
    return df
    

def drop(path):
    
    functions = [ 
        drop_single_action_users,
        drop_single_and_excessive_step_sessions,
        drop_duplicate_steps_in_session
    ];
    
    df = pd.read_csv(path,sep=",")
    origin_len = len(df)
    
    for fun in functions:
        prev_len = len(df);
        df = fun(df);
        cur_len = len(df);
        print(f"{fun.__name__}: Dropped {prev_len - cur_len} records.")
    
    print(f"{path.name} - Previously {origin_len}, now {cur_len} (Dropped {origin_len - cur_len} in total).");
    return df


def explode_multivalue_attributes(df,columns):
    print('columns to be exploded')
    print(columns);
    for col in columns:
        print(col)
        df[col] = df[col].apply(lambda x: x.split('|') if x != None else x)
        
    return df

def add_first_interaction_column(df):
    first_interaction = df[df['reference'].apply(lambda x: x.isnumeric() if x != None else False)].groupby('reference').agg({'timestamp':'min'}).reset_index().rename(columns={'timestamp':'first_interaction'})
    df = df.merge(first_interaction,on=['reference'],how='left').reset_index(drop=True)

    return df


def encode_categorical(df,columns):
    for col in columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    
    return df

def one_hot_encode(df,columns):
    print('columns one hot')
    print(columns);
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
        add_first_interaction_column,
        partial(encode_categorical, columns=['user_id','session_id', 'action_type', 'platform', 'city']),
        partial(one_hot_encode, columns=['device']),
        remove_step_column
    ];
    
    for fun in functions:
        df = fun(df);
    
    return df

def generate_statistics_from_parquet(data_location, stats_options=options.StatsOptions()):
    
    output_path = os.path.join(tempfile.mkdtemp(), "data_stats.tfrecord")
    output_dir_path = os.path.dirname(output_path)
    
    if not tf.io.gfile.exists(output_dir_path):
        tf.io.gfile.makedirs(output_dir_path)

    def decode_parquet(table: pa.lib.Table):
        return pa.RecordBatch.from_arrays(
            [ToSingletonListArray(col.chunks[0]) for col in table.columns],
            table.column_names,
        )

    with beam.Pipeline(options=None) as p:
        _ = (
            p
            | "ReadData" >> beam.io.parquetio.ReadFromParquetBatched(str(data_location))
            | "DecodeParquet" >> beam.Map(decode_parquet)
            | "GenerateStatistics" >> stats_api.GenerateStatistics(stats_options)
            | "WriteStatsOutput" >> stats_api.WriteStatisticsToTFRecord(output_path)
        )
    return stats_util.load_statistics(output_path)


def generate_schema_from_parquet(input_file,output_path):
    schema = tfdv.infer_schema(generate_statistics_from_parquet(input_file))
    
    # until https://github.com/NVIDIA-Merlin/Transformers4Rec/issues/357 is fixed
    schema_json = json.loads(json_format.MessageToJson(schema))

    def add_annotation(x):
        x['annotation'] = {}
        return x

    schema_json['feature'] = list(map(lambda x: add_annotation(x), schema_json['feature']))
    
    file_io.write_string_to_file(output_path, json.dumps(schema_json))