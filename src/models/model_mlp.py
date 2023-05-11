from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint

from _helpers import functions as hf


class ModelMLP():
    """
    Model class using neural network
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        df_impressions = hf.load_preprocessed_dataset('train')
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        X = df_impressions[self.params['features']]
        y = df_impressions.is_clicked
        
        # Perform feature scaling
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Define the parameter grid for randomized search
        param_grid = {
            'hidden_layer_sizes': randint(64, 256),
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01]
        }
        
        # Train the model
        model = MLPRegressor(random_state=42)
        
        # Perform randomized search
        randomized_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, verbose=True)
        randomized_search.fit(X_train, y_train)
        
        self.model = randomized_search.best_estimator_
    
    def predict(self, *args, **kwargs):
        df_impressions = hf.load_preprocessed_dataset('test')
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        X = df_impressions[self.params['features']]
        
        # Perform feature scaling using the fitted StandardScaler from the training data
        X = self.scaler.transform(X)
        
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
