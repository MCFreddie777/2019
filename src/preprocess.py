import pandas as pd

from _helpers import constants
from _helpers import functions as hf
from _helpers.preprocess import preprocess, MetaPreprocesser
from drop import main as drop


def split_df_into_session_chunks(df, chunk_size):
    """
    Gets unique user - session pairs and splits dataframe into chunks of N user - session pairs in chunk (N == chunk_size)
    """
    unique_sessions = df[['user_id', 'session_id']].drop_duplicates()['session_id']
    return [unique_sessions[i:i + chunk_size] for i in range(0, len(unique_sessions), chunk_size)]


def preprocess_and_save_chunks(df, type):
    chunk_size = round(float(hf.get_env('PREPROCESS_CHUNK_SIZE', 1e5)))
    chunks = split_df_into_session_chunks(df, chunk_size)
    
    df_meta_preprocessed = MetaPreprocesser(constants.METADATA).df_meta
    
    # Iterate over the chunks
    for i, chunk in enumerate(chunks):
        df_chunk = df[df['session_id'].isin(chunk)]
        processed_chunk = preprocess(df_chunk, df_meta_preprocessed)
        
        processed_chunk.to_parquet(constants.PREPROCESSED(i, type), index=False)
        print(f"Chunk {i} saved to {constants.PREPROCESSED(i, type)}.")


def main():
    """
    This function prepares test and train data for training
    """
    try:
        hf.require_files([constants.DROPPED_TRAIN, constants.TEST])
    except FileNotFoundError:
        drop()
    
    # Preprocess train
    print(f"Reading {constants.DROPPED_TRAIN}...")
    df_train = pd.read_parquet(constants.DROPPED_TRAIN)
    preprocess_and_save_chunks(df_train, 'train')
    
    # Preprocess test
    print(f"Reading {constants.TEST}...")
    df_test = hf.reduce_mem_usage(pd.read_csv(constants.TEST))
    preprocess_and_save_chunks(df_test, 'test')
    
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
