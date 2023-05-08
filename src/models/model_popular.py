from _helpers import constants
from _helpers import functions as hf


class ModelPopular():
    """
    Model based on total number of clicks per item.
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, df):
        """Get number of clicks that each item received in the df."""
        
        df_cols = df.copy().loc[df["action_type"].isin(constants.ITEM_REFERENCE_ACTION_TYPE_COLS), :]

        # Get number of user interactions with specific item
        df_user_item_interactions = (
            df_cols
            .groupby(['user_id', 'reference'])
            .size()
            .reset_index(name='user_item_interactions')
        )
        
        # Get global number of interactions with the specific item
        df_item_interactions = (
            df_cols
            .groupby('reference')
            ['user_id']
            .nunique()
            .reset_index(name="item_interactions")
        )
        
        self.df_user_item_pop = df_user_item_interactions.merge(df_item_interactions,on="reference").rename(columns={'reference':'item'})
    
    def predict(self, df):
        features = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions']
        df_cols = df.copy().loc[:, features]
        
        df_target = hf.get_target_rows(df_cols.copy())
        
        # Explode the impressions into separate rows
        df_impressions = (
            hf.explode(df_target, ["impressions"])
            .rename(columns={"impressions": "impressed_item"})
        )
        
        # Merge with preprocessed dataset from fit function
        df_impressions = (
            df_impressions
            .merge(
                self.df_user_item_pop,
                left_on=['user_id',"impressed_item"],
                right_on=['user_id',"item"],
                how="left"
            )
        )
        
        # Sort impressions by user_item interactions and global interactions
        df_rec = (
            df_impressions
            .sort_values(
                by=["user_id", "session_id", "timestamp", "step",'user_item_interactions','item_interactions'],
                ascending=[True, True, True, True, False,False],
                na_position='last'
            )
            .groupby(["user_id", "session_id", "timestamp", "step"])["impressed_item"]
            .apply(lambda x: ' '.join(x))
            .to_frame()
            .reset_index()
            .rename(columns={'impressed_item': 'item_recommendations'})
        )
        
        return df_rec
