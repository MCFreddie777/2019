import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import joblib

from _helpers import constants
from _helpers import functions as hf


class ModelLogisticRegression():
    """
    Model class for the logistic regression model.
    """
    params = {}
    
    def __get_features_and_labels(self, filename):
        df_impressions = pd.read_parquet(filename)
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        X = df_impressions[self.params['features']]
        y = df_impressions.is_clicked
        
        return X, y
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        """Train the logistic regression model."""
        wandb = kwargs['wandb']
        
        param_grid = {
            'solver': ['lbfgs', 'liblinear'],
            'C': [0.1, 1.0, 10.0, 100.0]
        }
        
        logreg_clf = LogisticRegression(max_iter=100, tol=1e-11, warm_start=True, verbose=True)
        
        train_chunks = hf.get_preprocessed_dataset_chunks('train')
        
        # Perform randomized search on first chunk
        X, y = self.__get_features_and_labels(train_chunks[0])
        
        randomized_search = RandomizedSearchCV(
            estimator=logreg_clf,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
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
        self.logreg = randomized_search.best_estimator_
        
        # Partially fit the best estimator on subsequent chunks (warm_start=True)
        for chunk_filename in train_chunks:
            X, y = self.__get_features_and_labels(chunk_filename)
            self.logreg.fit(X, y)
        
        # Persist model in file
        joblib.dump(self.logreg, constants.MODEL_DIR / f'{self.params["model"]}_{self.params["timestamp"]}')
    
    def predict(self, *args, **kwargs):
        """Calculate click probability based on trained logistic regression model."""
        
        df_impressions = hf.load_preprocessed_dataset('test')
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        df_impressions.loc[:, "click_probability"] = (
            self
            .logreg
            .predict_proba(df_impressions[self.params['features']])[:, 1]
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
