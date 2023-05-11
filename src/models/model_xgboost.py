import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from _helpers import functions as hf


class ModelXGBoost():
    """
    Model class using XGBoost library.
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        """Train the XGBoost model."""
        
        df_impressions = hf.load_preprocessed_dataset('train')
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        X = df_impressions[self.params['features']]
        y = df_impressions.is_clicked
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300]
        }
        
        xgb_clf = xgb.XGBClassifier(verbosity=1)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_grid, n_iter=10, cv=3, verbose=True) \
            .fit(X, y)
        
        self.xgb = random_search.best_estimator_
    
    def predict(self, *args, **kwargs):
        """Calculate click probability based on XGBoost prediction probability."""
        
        df_impressions = hf.load_preprocessed_dataset('test')
        
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
