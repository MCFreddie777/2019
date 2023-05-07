import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def explode_mult(df_in, col_list):
    """Explode each column in col_list into multiple rows."""
    
    df = df_in.copy()
    
    for col in col_list:
        df.loc[:, col] = df.loc[:, col].str.split("|")
    
    df_out = pd.DataFrame(
        {col: np.repeat(df[col].to_numpy(),
                        df[col_list[0]].str.len())
         for col in df.columns.drop(col_list)}
    )
    
    for col in col_list:
        df_out.loc[:, col] = np.concatenate(df.loc[:, col].to_numpy())
    
    return df_out

def preprocess(df):
    """Build features for the lightGBM and logistic regression model."""
    
    # Select columns that are of interest for this method
    cols = ['user_id', 'session_id', 'timestamp', 'step',
            'action_type', 'reference', 'impressions', 'prices']
    df_cols = df.loc[:, cols]
    
    # We are only interested in action types, for which the reference is an item ID
    item_interactions = [
        'clickout item', 'interaction item deals', 'interaction item image',
        'interaction item info', 'interaction item rating', 'search for item'
    ]
    df_actions = (
        df_cols
        .loc[df_cols.action_type.isin(item_interactions), :]
        .copy()
        .rename(columns={'reference': 'referenced_item'})
    )
    
    # Clean of instances that have no reference
    idx_rm = (df_actions.action_type != "clickout item") & (df_actions.referenced_item.isna())
    df_actions = df_actions[~idx_rm]
    
    # Get item ID of previous interaction of a user in a session
    df_actions.loc[:, "previous_item"] = (
        df_actions
        .sort_values(by=["user_id", "session_id", "timestamp", "step"],
                     ascending=[True, True, True, True])
        .groupby(["user_id"])["referenced_item"]
        .shift(1)
    )
    
    # Combine the impressions and item column, they both contain item IDs
    # and we can expand the impression lists in the next step to get the total
    # interaction count for an item
    df_actions.loc[:, "interacted_item"] = np.where(
        df_actions.impressions.isna(),
        df_actions.referenced_item,
        df_actions.impressions
    )
    df_actions = df_actions.drop(columns="impressions")
    
    # Price array expansion will get easier without NAs
    df_actions.loc[:, "prices"] = np.where(
        df_actions.prices.isna(),
        "",
        df_actions.prices
    )
    
    # Convert pipe separated lists into columns
    df_items = explode_mult(df_actions, ["interacted_item", "prices"]).copy()
    
    # Feature: Number of previous interactions with an item
    df_items.loc[:, "interaction_count"] = (
        df_items
        .groupby(["user_id", "interacted_item"])
        .cumcount()
    )
    
    # Reduce to impression level again
    df_impressions = (
        df_items[df_items.action_type == "clickout item"]
        .copy()
        .drop(columns="action_type")
        .rename(columns={"interacted_item": "impressed_item"})
    )
    
    # Feature: Position of item in the original list.
    # Items are in original order after the explode for each index
    df_impressions.loc[:, "position"] = (
            df_impressions
            .groupby(["user_id", "session_id", "timestamp", "step"])
            .cumcount() + 1
    )
    
    # Feature: Is the impressed item the last interacted item
    df_impressions.loc[:, "is_last_interacted"] = (
            df_impressions["previous_item"] == df_impressions["impressed_item"]
    ).astype(int)
    
    df_impressions.loc[:, "prices"] = df_impressions.prices.astype(int)
    
    return_cols = [
        "user_id",
        "session_id",
        "timestamp",
        "step",
        "position",
        "prices",
        "interaction_count",
        "is_last_interacted",
        "referenced_item",
        "impressed_item",
    ]
    
    df_return = df_impressions[return_cols]
    
    return df_return


def build_features(df):
    """Build features for the lightGBM and logistic regression model."""
    
    # Select columns that are of interest for this method
    cols = ['user_id', 'session_id', 'timestamp', 'step',
            'action_type', 'reference', 'impressions', 'prices']
    df_cols = df.loc[:, cols]
    
    # We are only interested in action types, for which the reference is an item ID
    item_interactions = [
        'clickout item', 'interaction item deals', 'interaction item image',
        'interaction item info', 'interaction item rating', 'search for item'
    ]
    df_actions = (
        df_cols
        .loc[df_cols.action_type.isin(item_interactions), :]
        .copy()
        .rename(columns={'reference': 'referenced_item'})
    )
    
    # Clean of instances that have no reference
    idx_rm = (df_actions.action_type != "clickout item") & (df_actions.referenced_item.isna())
    df_actions = df_actions[~idx_rm]
    
    # Get item ID of previous interaction of a user in a session
    df_actions.loc[:, "previous_item"] = (
        df_actions
        .sort_values(by=["user_id", "session_id", "timestamp", "step"],
                     ascending=[True, True, True, True])
        .groupby(["user_id"])["referenced_item"]
        .shift(1)
    )
    
    # Combine the impressions and item column, they both contain item IDs
    # and we can expand the impression lists in the next step to get the total
    # interaction count for an item
    df_actions.loc[:, "interacted_item"] = np.where(
        df_actions.impressions.isna(),
        df_actions.referenced_item,
        df_actions.impressions
    )
    df_actions = df_actions.drop(columns="impressions")
    
    # Price array expansion will get easier without NAs
    df_actions.loc[:, "prices"] = np.where(
        df_actions.prices.isna(),
        "",
        df_actions.prices
    )
    
    # Convert pipe separated lists into columns
    df_items = explode_mult(df_actions, ["interacted_item", "prices"]).copy()
    
    # Feature: Number of previous interactions with an item
    df_items.loc[:, "interaction_count"] = (
        df_items
        .groupby(["user_id", "interacted_item"])
        .cumcount()
    )
    
    # Reduce to impression level again
    df_impressions = (
        df_items[df_items.action_type == "clickout item"]
        .copy()
        .drop(columns="action_type")
        .rename(columns={"interacted_item": "impressed_item"})
    )
    
    # Feature: Position of item in the original list.
    # Items are in original order after the explode for each index
    df_impressions.loc[:, "position"] = (
            df_impressions
            .groupby(["user_id", "session_id", "timestamp", "step"])
            .cumcount() + 1
    )
    
    # Feature: Is the impressed item the last interacted item
    df_impressions.loc[:, "is_last_interacted"] = (
            df_impressions["previous_item"] == df_impressions["impressed_item"]
    ).astype(int)
    
    df_impressions.loc[:, "prices"] = df_impressions.prices.astype(int)
    
    return_cols = [
        "user_id",
        "session_id",
        "timestamp",
        "step",
        "position",
        "prices",
        "interaction_count",
        "is_last_interacted",
        "referenced_item",
        "impressed_item",
    ]
    
    df_return = df_impressions[return_cols]
    
    return df_return


class ModelLogisticRegression():
    """
    Model class for the logistic regression model.
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, df):
        """Train the logistic regression model."""
        
        df_impressions = preprocess(df)
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        features = [
            "position",
            "prices",
            "interaction_count",
            "is_last_interacted",
        ]
        
        X = df_impressions[features]
        y = df_impressions.is_clicked
        
        self.logreg = LogisticRegression(solver="lbfgs", max_iter=100, tol=1e-11, C=1e10, verbose=True).fit(X, y)
    
    def predict(self, df):
        """Calculate click probability based on trained logistic regression model."""
        
        df_impressions = preprocess(df)
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        features = [
            "position",
            "prices",
            "interaction_count",
            "is_last_interacted"
        ]
        
        df_impressions.loc[:, "click_probability"] = (
            self
            .logreg
            .predict_proba(df_impressions[features])[:, 1]
        )
        
        df_rec = (
            df_impressions
            .sort_values(by=["user_id", "session_id", "timestamp", "step", 'click_probability'],
                         ascending=[True, True, True, True, False])
            .groupby(["user_id", "session_id", "timestamp", "step"])["impressed_item"]
            .apply(lambda x: ' '.join(x))
            .to_frame()
            .reset_index()
            .rename(columns={'impressed_item': 'item_recommendations'})
        )
        
        return df_rec
