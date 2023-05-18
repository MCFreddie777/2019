from functools import partial
import numpy as np
import pandas as pd

from . import constants
from . import functions as hf
from .functions import verbose_print


def __select_only_reference_action_columns(df):
    """
    Select only those action_type rows, which contain item_id in reference column
    """
    df2 = df[
        df.action_type.isin(constants.ITEM_REFERENCE_ACTION_TYPE_COLS)
    ] \
        .copy() \
        .rename(columns={'reference': 'referenced_item'})
    
    # Just in case, if there are any reference action columns without reference present, drop them
    df2 = df2[~((df2.action_type != "clickout item") & (df2['referenced_item'].isna()))]
    
    return df2


def __add_previous_item_column(df):
    """
    Get item id of previous interaction of a user in a session
    """
    
    df2 = df.copy()
    df2.insert(
        loc=df2.columns.get_loc("referenced_item") + 1,  # Insert previous item after reference column
        column='previous_item',
        value=df2.sort_values(
            by=["user_id", "session_id", "timestamp", "step"],
            ascending=[True, True, True, True]
        )
        .groupby(["user_id"])["referenced_item"]
        .shift(1)
    )
    
    return df2


def __explode_impressions_prices_columns(df):
    """
    Explodes impressions and its prices into separate rows
    """
    
    df2 = df.copy()
    
    # Fill interacted_item column either with impressions (clickout item) or reference (other action_types)
    df2.insert(
        loc=df2.columns.get_loc("referenced_item") + 1,  # Insert previous item after reference column
        column='interacted_item',
        value=np.where(
            df2.impressions.isna(),
            df2.referenced_item,
            df2.impressions
        )
    )
    
    df2.loc[:, "prices"] = np.where(
        df2.prices.isna(),
        "",
        df2.prices
    )
    
    df2 = hf.explode(df2, ['interacted_item', 'prices'])
    
    # Rename prices since it represents single value now
    df2 = df2.rename(columns={'prices': 'price'})
    
    # Don't need this column anymore as it
    df2 = df2.drop(columns="impressions")
    
    # Reorder columns, visually
    df2 = hf.reorder_column(df2, df2.columns.get_loc("interacted_item"), df2.columns.get_loc("referenced_item") + 1)
    df2 = hf.reorder_column(df2, df2.columns.get_loc("price"), df2.columns.get_loc("interacted_item") + 1)
    
    return df2


def __add_user_interacted_item_interaction_count_column(df):
    df2 = df.copy()
    
    df2.insert(
        loc=df2.columns.get_loc("interacted_item") + 1,  # Insert previous item after reference column
        column='user_interacted_item_interaction_count',
        value=(
            df
            .groupby(["user_id", "interacted_item"])
            .cumcount()
        )
    )
    
    return df2


def __add_last_interacted_column(df):
    """
    User modelling
    Adds is_last_interacted column which presents whether impressed_item the last interacted_item
    
    At this point, in impressed_item represents any of the impressions during the clickout
    previous_item column represents other item from another action_type, e.g interaction_with_image (already dropped columns)
    
    User is more likely to clickout the item (therefore have it in the referenced_item), if he previously interacted with it in session
    """
    df2 = df.copy()
    
    df2.insert(
        loc=df2.columns.get_loc("user_impressed_item_interaction_count") + 1,
        # Insert previous item after reference column
        column='is_last_interacted',
        value=(df2["previous_item"] == df2["impressed_item"]).astype(int)
    )
    
    return df2


def __narrow_to_clickouts(df):
    return df[df.action_type == "clickout item"] \
        .copy() \
        .drop(columns="action_type") \
        .rename(columns={"interacted_item": "impressed_item"}) \
        .rename(columns={"user_interacted_item_interaction_count": "user_impressed_item_interaction_count"})


def __add_impressed_item_position_column(df):
    """
    This function labels each impression with number which represents its order in impression list
    """
    
    target_rows = df['action_type'] == 'clickout item'
    df.loc[target_rows, 'impressed_item_position'] = df[target_rows].groupby(['session_id', 'step']).cumcount() + 1
    
    return df


def __add_relative_impressed_item_position_column(df):
    """
    Inserts relative position of the impression in exploded dataframe in current session
    First impression is 0, last is 1
    """
    df2 = df.copy()
    
    # Group dataframe by all impressions and calculate (position out of all impressions)
    grouped = df2 \
        .sort_values(by=['user_id', 'session_id', 'timestamp', 'step', 'impressed_item_position']) \
        .groupby(['user_id', 'session_id', 'timestamp', 'step'])
    
    df2.insert(
        loc=df2.columns.get_loc("impressed_item_position") + 1,  # Insert after column
        column='relative_impressed_item_position',
        value=grouped.cumcount() / (grouped['impressed_item_position'].transform('count') - 1)
    )
    
    # For single impression clickouts, the result above is NaN because division by zero
    df2["relative_impressed_item_position"].fillna(1, inplace=True)
    
    return df2


def __add_mean_price_column(df):
    df_mean_price = df \
        .groupby(['user_id', 'session_id', 'timestamp', 'step']) \
        .agg(mean_impressions_price=('price', 'mean')) \
        .reset_index()
    
    df2 = df_mean_price.merge(
        df,
        on=['user_id', 'session_id', 'step', 'timestamp'],
    )
    
    df2['price_above_impression_mean'] = (df2['price'] > df2['mean_impressions_price']).astype(int)
    
    return df2


def __add_relative_price_column(df):
    """
    This function calculates relative price of the impression in current session
    """
    df2 = df.copy()
    
    df2.insert(
        loc=df2.columns.get_loc("price") + 1,  # Insert after price column
        column='relative_price',
        value=(
            df2
            .groupby(['user_id', 'session_id', 'timestamp', 'step'])['price']
            .transform(lambda x: x / x.iloc[0])
        )
    )
    
    return df2


def __encode_cat_columns(df, columns):
    df2 = df.copy()
    
    for cat in columns:
        df2[cat] = df2[cat].astype('category').cat.codes
    
    return df2


class MetaPreprocesser():
    """
    Class for processing metadata
    Stateful, so it can be used for preprocessing multiple times
    """
    
    def __init__(self, meta_file=constants.METADATA):
        self.df_meta = pd.read_csv(meta_file, dtype={'item_id': str})
        
        # Split into array
        self.df_meta.loc[:, 'properties'] = self.df_meta.loc[:, 'properties'].str.split("|")
        
        rating_map = {
            'Satisfactory Rating': 1,
            'Good Rating': 2,
            'Very Good Rating': 3,
            'Excellent Rating': 4,
        }
        
        # Properties contain multiple ratings, all of those which apply, we need to find a maximum
        self.df_meta['impressed_item_rating'] = self.df_meta['properties'].apply(
            lambda x: max([rating_map[key] for key in x if key in rating_map], default=None))


def __add_rating_column(df, df_meta_preprocessed=MetaPreprocesser().df_meta):
    df2 = df.merge(
        df_meta_preprocessed[['item_id', 'impressed_item_rating']],
        left_on='impressed_item',
        right_on='item_id',
        how='left'
    )
    
    # Left join, fill empty ratings (impressed_item wasn't in df_meta) with mean
    df2['impressed_item_rating'] = df2['impressed_item_rating'].fillna(2)
    
    df2 = df2.drop(columns='item_id')
    
    return df2


def __cast_price_to_float(df):
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df


def __add_one_of_n_cheapest_column(df, n):
    """
    Adds column which contains the information whether the current impression is one of n cheapest in current session
    """
    df['is_one_of_n_cheapest'] = (
            df.groupby(['user_id', 'session_id', 'timestamp', 'step'])
            ['price']
            .rank(ascending=True, method='dense') <= n
    ).astype(int)
    
    return df


def __add_user_interacted_with_n_cheapest(df):
    """
    Adds column which contains the information whether user interacted previously with one of n cheapest items
    """
    df['user_interacted_with_n_cheapest'] = df.groupby('user_id') \
        ['is_one_of_n_cheapest'] \
        .transform(lambda x: x.shift().cumsum().fillna(0).astype(int))
    
    return df


def __add_one_of_the_top_n_column(df, n):
    """
    Adds column which contains the information whether the current impression is one of top n impressions
    """
    df['is_one_of_the_top_n'] = (df['impressed_item_position'] <= n).astype(int)
    
    return df


def __add_user_interacted_with_top_n_column(df):
    """
    Adds column which contains the information whether user interacted previously with one of the top n items
    """
    df['user_interacted_with_top_n'] = df.groupby('user_id') \
        ['is_one_of_the_top_n'] \
        .transform(lambda x: x.shift().cumsum().fillna(0).astype(int))
    
    return df


def __collect_features(df):
    features = [
        "user_id",
        "session_id",
        "timestamp",
        "step",
        "referenced_item",
        "impressed_item",
        "impressed_item_position",
        "relative_impressed_item_position",
        "impressed_item_rating",
        "user_impressed_item_interaction_count",
        "user_interacted_with_n_cheapest",
        "user_interacted_with_top_n",
        "is_last_interacted",
        "is_one_of_n_cheapest",
        "is_one_of_the_top_n",
        "price",
        "relative_price",
        "price_above_impression_mean",
        "device",
        "platform",
        "city",
    ]
    
    return df[features]


def preprocess(df, df_meta_preprocessed):
    """
    Function that preprocesses the dataset
    
    :param df: Dataframe to be preprocessed
    :return: New dataframe
    """
    
    # Pipeline of preprocess functions
    functions = [
        partial(__select_only_reference_action_columns),
        partial(__add_previous_item_column),
        partial(__explode_impressions_prices_columns),
        partial(__add_user_interacted_item_interaction_count_column),
        partial(__add_impressed_item_position_column),
        partial(__add_one_of_the_top_n_column, n=3),
        partial(__add_user_interacted_with_top_n_column),
        partial(__cast_price_to_float),
        partial(__add_one_of_n_cheapest_column, n=3),
        partial(__add_user_interacted_with_n_cheapest),
        partial(__narrow_to_clickouts),
        partial(__add_mean_price_column),
        partial(__add_relative_price_column),
        partial(__add_relative_impressed_item_position_column),
        partial(__add_last_interacted_column),
        partial(__encode_cat_columns, columns=['device', 'platform', 'city']),
        partial(__add_rating_column, df_meta_preprocessed=df_meta_preprocessed),
        partial(__encode_cat_columns, columns=['impressed_item_rating']),
        partial(__collect_features),
    ]
    
    for fun in functions:
        verbose_print(f"Running {fun.__name__ if hasattr(fun, '__name__') else fun.func.__name__}...")
        df = fun(df)
    
    return df
