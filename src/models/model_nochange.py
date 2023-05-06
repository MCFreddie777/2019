from _helpers import functions as f

class ModelNoChange():
    """
    Model class for the model without change based on displayed items list.
    """
    params = {}
    
    def update(self,params):
        self.params = params
    
    def fit(self, _):
        pass
    
    def predict(self, df):
        
        df_target = f.get_target_rows(df.copy())
        
        df_target["item_recommendations"] = (
            df_target
            .apply(lambda x: x.impressions.replace("|", " "), axis=1)
        )
        
        cols_rec = ["user_id", "session_id", "timestamp", "step", "item_recommendations"]
        df_rec = df_target.loc[:, cols_rec]
        
        return df_rec
