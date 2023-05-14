import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import joblib

from _helpers import constants
from _helpers import functions as hf


class ModelXGBoost():
    """
    Model class using XGBoost library.
    """
    params = {}
    
    def __get_features_and_labels(self, filename):
        df = pd.read_parquet(filename)
        
        df.loc[:, "is_clicked"] = (
                df["referenced_item"] == df["impressed_item"]
        ).astype(int)
        
        X = df[self.params['features']]
        y = df.is_clicked
        
        return X, y
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        """Train the XGBoost model."""
        wandb = kwargs['wandb']
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'n_estimators': [100, 200, 300]
        }
        
        xgb_clf = xgb.XGBClassifier(verbosity=1)
        
        train_chunks = hf.get_preprocessed_dataset_chunks('train')
        
        # Perform randomized search on first chunk
        X, y = self.__get_features_and_labels(train_chunks[0])
        
        # Perform randomized search
        randomized_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            verbose=True
        )
        randomized_search.fit(X, y)
        
        # Save hyperparameter tuning results
        wandb.log({
            "hyperparameter_tuning": {
                "type": randomized_search.__class__,
                "best_score": randomized_search.best_score_,
                "best_params": randomized_search.best_params_,
                "best_estimator": str(randomized_search.best_estimator_)
            }
        })
        
        # Save best model
        self.xgb = randomized_search.best_estimator_
        
        # Partially fit the best estimator on subsequent chunks
        for i, chunk_filename in enumerate(train_chunks):
            X, y = self.__get_features_and_labels(chunk_filename)
            if i == 0:
                self.xgb.fit(X, y)
            else:
                self.xgb.fit(X, y, xgb_model=self.xgb)
        
        # Persist model in file
        joblib.dump(self.xgb, constants.MODEL_DIR / f'{self.params["model"]}_{self.params["timestamp"]}')
    
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
