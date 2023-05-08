import pandas as pd
from sklearn.linear_model import LogisticRegression

from _helpers.preprocess import preprocess
from _helpers import constants


def _get_preprocessed_from_file(type):
    train_file = constants.PREPROCESSED_TRAIN
    test_file = constants.PREPROCESSED_TEST
    
    if (constants.SUBSET is not None):
        train_file = constants.PREPROCESSED_SUBSET(constants.SUBSET, 'train')
        test_file = constants.PREPROCESSED_SUBSET(constants.SUBSET, 'test')
    
    target_file = train_file if type == 'train' else test_file
    
    return pd.read_parquet(target_file)


class ModelLogisticRegression():
    """
    Model class for the logistic regression model.
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, df):
        """Train the logistic regression model."""
        
        try:
            df_impressions = _get_preprocessed_from_file('train')
        except FileNotFoundError:
            df_impressions = preprocess(df)
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        X = df_impressions[self.params['features']]
        y = df_impressions.is_clicked
        
        self.logreg = LogisticRegression(solver="lbfgs", max_iter=100, tol=1e-11, C=1e10, verbose=True).fit(X, y)
    
    def predict(self, df):
        """Calculate click probability based on trained logistic regression model."""
        
        try:
            df_impressions = _get_preprocessed_from_file('test')
        except FileNotFoundError:
            df_impressions = preprocess(df)
        
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
