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
        
        features = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference']
        df_cols = df.copy().loc[:, features]
        
        df_item_clicks = (
            df_cols
            .loc[df_cols["action_type"] == "clickout item", :]
            .groupby("reference")
            .size()
            .reset_index(name="n_clicks")
            .rename(columns={"reference": "item"})
        )
        
        self.df_pop = df_item_clicks
    
    def predict(self, df):
        features = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions']
        df_cols = df.copy().loc[:, features]
        
        df_target = hf.get_target_rows(df_cols.copy())
        
        df_impressions = (
            hf.explode(df_target, "impressions")
            .rename(columns={"impressions": "impressed_item"})
        )
        
        df_impressions = (
            df_impressions
            .merge(
                self.df_pop,
                left_on="impressed_item",
                right_on="item",
                how="left"
            )
        )
        
        df_rec = (
            df_impressions
            .sort_values(
                by=["user_id", "session_id", "timestamp", "step", 'n_clicks'],
                ascending=[True, True, True, True, False]
            )
            .groupby(["user_id", "session_id", "timestamp", "step"])["impressed_item"]
            .apply(lambda x: ' '.join(x))
            .to_frame()
            .reset_index()
            .rename(columns={'impressed_item': 'item_recommendations'})
        )
        
        return df_rec
