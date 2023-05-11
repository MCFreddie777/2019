from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pandas as pd

from _helpers import functions as hf


class ModelMLP():
    """
    Model class using neural network
    """
    params = {}
    
    def __transform_for_fitting(self, filename):
        df = pd.read_parquet(filename)
        
        df.loc[:, "is_clicked"] = (df["referenced_item"] == df["impressed_item"]).astype(int)
        
        X = df[self.params['features']]
        y = df.is_clicked
        
        return X, y
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        # Define the parameter grid for randomized search
        param_grid = {
            'hidden_layer_sizes': randint(64, 256),
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01]
        }
        
        train_chunks = hf.get_preprocessed_dataset_chunks('train')
        
        # We will optimalize hypterparameters on first chunk
        X, y = self.__transform_for_fitting(train_chunks[0])
        
        # Define model
        model = MLPRegressor(random_state=42, verbose=True)
        
        # Perform randomized search
        randomized_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=5,
            cv=3,
            verbose=True
        )
        randomized_search.fit(X, y)
        
        self.model = randomized_search.best_estimator_
        
        for i, chunk_filename in enumerate(train_chunks):
            X, y = self.__transform_for_fitting(chunk_filename)
            self.model.partial_fit(X, y)
    
    def predict(self, *args, **kwargs):
        df_impressions = hf.load_preprocessed_dataset('test')
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        X = df_impressions[self.params['features']]
        
        # Make predictions using the trained model
        df_impressions.loc[:, "click_probability"] = (self.model.predict(X))
        
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
