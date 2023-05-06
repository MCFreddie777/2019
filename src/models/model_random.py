import random

from _helpers import functions as f

class ModelRandom():
    """This model randomly sorts the impressions list."""
    params = {
        'seed': 123
    }
    
    def update(self,params):
        self.params = params
    
    def fit(self, _):
        pass
    
    def predict(self, df):
        
        df_target = f.get_target_rows(df.copy())
        
        random.seed(self.params['seed'])
    
        df_target.loc[:, "item_recommendations"] = (
            df_target
            .loc[:, "impressions"].str.split("|")
            .map(lambda x: sorted(x, key=lambda k: random.random()))
            .map(lambda arr: ' '.join(arr))
        )
        
        cols_rec = ["user_id", "session_id", "timestamp", "step", "item_recommendations"]
        df_rec = df_target.loc[:, cols_rec]

        return df_rec
