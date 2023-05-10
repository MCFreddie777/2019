import xgboost as xgb

from _helpers.preprocess import preprocess
from _helpers import functions as hf


class ModelXGBoost():
    """
    Model class using XGBoost library.
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, df):
        """Train the XGBoost model."""
        
        try:
            df_impressions = hf.load_preprocessed_dataset('train')
        except FileNotFoundError:
            df_impressions = preprocess(df)
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        X = df_impressions[self.params['features']]
        y = df_impressions.is_clicked
        
        self.xgb = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1).fit(X, y)
    
    def predict(self, df):
        """Calculate click probability based on XGBoost prediction probability."""
        
        try:
            df_impressions = hf.load_preprocessed_dataset('test')
        except FileNotFoundError:
            df_impressions = preprocess(df)
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        X = df_impressions[self.params['features']]
        
        df_impressions.loc[:, "click_probability"] = (
            self
            .xgb
            .predict_proba(X=X)[:, 1]
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
