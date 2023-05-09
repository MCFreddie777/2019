import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

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


class ModelNeural():
    """
    Model class using neural network
    """
    params = {}
    
    def update(self, params):
        self.params = params
    
    def fit(self, df):
        
        try:
            df_impressions = _get_preprocessed_from_file('train')
        except FileNotFoundError:
            df_impressions = preprocess(df)
        
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
        
        # Define the parameter grid
        param_grid = {
            'epochs': [10, 20],  # Number of epochs to train the model
            'batch_size': [32, 64]  # Batch size for training
        }
        
        # Create the neural network model
        model = Sequential()
        model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Perform grid search using cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        # Get the best model from grid search
        self.model = grid_search.best_estimator_
        
        # Train the best model on the full training data
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.model.get_params()['epochs'],
            batch_size=self.model.get_params()['batch_size']
        )
        
    
    def predict(self, df):
        
        try:
            df_impressions = _get_preprocessed_from_file('test')
        except FileNotFoundError:
            df_impressions = preprocess(df)
        
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
