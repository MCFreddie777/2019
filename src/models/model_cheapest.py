from _helpers import functions as hf


class ModelCheapest():
    """This model selects cheapest from the impressions list."""
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, _):
        pass
    
    def predict(self, df):
        df_target = hf.get_target_rows(df.copy())
        
        # Explode the impression-price pairs into separate rows
        df_impressions = (
            hf.explode(df_target, ["impressions", "prices"])
            .rename(columns={"impressions": "impressed_item", 'prices': 'price'})
        )
        
        # Sort impressions by user_item interactions and global interactions
        df_rec = (
            df_impressions
            .sort_values(
                by=["user_id", "session_id", "timestamp", "step",'price'],
                ascending=True,
                na_position='last'
            )
            .groupby(["user_id", "session_id", "timestamp", "step"])["impressed_item"]
                .apply(lambda x: ' '.join(x))
                .to_frame()
                .reset_index()
                .rename(columns={'impressed_item': 'item_recommendations'})
        )
        
        return df_rec
