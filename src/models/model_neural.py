import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Reshape, Input, Dropout
from scikeras.wrappers import KerasClassifier
import numpy as np
import joblib

from _helpers import constants
from _helpers import functions as hf


class ModelNeural():
    """
    Model class using neural network
    """
    params = {}
    
    def __create_model(
            self,
            input_shape,
            activation=('relu', 'relu'),
            optimizer='adam',
            neurons=(64, 32),
            loss='binary_crossentropy'
    ):
        model = Sequential()
        
        # model.add(Input(shape=input_shape))
        # TODO Add embedding
        
        model.add(SimpleRNN(units=neurons[0], input_shape=input_shape, activation=activation[0]))
        model.add(Dense(units=neurons[1], activation=activation[1]))
        # model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss=loss, optimizer=optimizer)
        
        return model
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        wandb = kwargs['wandb']
        
        df_impressions = hf.load_preprocessed_dataset('train')
        
        df_impressions.loc[:, "is_clicked"] = (
                df_impressions["referenced_item"] == df_impressions["impressed_item"]
        ).astype(int)
        
        X = df_impressions[self.params['features']]
        y = df_impressions.is_clicked
        
        # TODO try scaling
        # Perform feature scaling
        # self.scaler = StandardScaler()
        # X = self.scaler.fit_transform(X)
        
        # TODO split by files
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        params = {
            'epochs': 1,
            "batch_size": 32,
            'activation': ['relu', 'relu'],
            'optimizer': 'adam',
            'neurons': (8, 4),  # (64,32)
            'loss': 'mean_squared_error'
        }
        
        # Define the parameter grid
        # param_grid = {
        #     'epochs': [10, 20],
        #     'batch_size': [32, 64],
        #     'neurons': [(128, 64), (64, 32), (32, 16)],
        #     'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
        #     'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
        #     'learn_rate': [.001, 0.01, 0.1, 0.2, 0.3],
        #     'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        # 'loss': ['binary_crossentropy', 'mean_squared_error']
        # }
        
        # Create the neural network model
        model = self.__create_model(
            input_shape=(X.shape[1], 1),
            neurons=params['neurons'],
            activation=params['activation'],
            optimizer=params['optimizer'],
            loss=params['loss']
        )  # KerasClassifier(model=self.__create_model, input_shape=(X.shape[1]))
        
        # Perform randomized search
        # randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, verbose=True)
        # randomized_search.fit(X_train, y_train)
        
        # Save hyperparameter tuning results
        wandb.log({
            "hyperparameter_tuning": params,  # {
            # "type": randomized_search.__class__,
            # "best_score": randomized_search.best_score_,
            # "best_params": randomized_search.best_params_,
            # "best_estimator": str(randomized_search.best_estimator_)
            # }
        })
        
        # Get the best model from grid search
        self.model = model  # randomized_search.best_estimator_
        
        # Train the best model on the full training data
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],  # self.model.get_params()['epochs'],
            batch_size=params['batch_size']  # self.model.get_params()['batch_size']
        )
        
        # Persist model in file
        joblib.dump(self.model, constants.MODEL_DIR / f'{self.params["model"]}_{self.params["timestamp"]}')
    
    def predict(self, *args, **kwargs):
        df_impressions = hf.load_preprocessed_dataset('test')
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        X = df_impressions[self.params['features']]
        
        # Perform feature scaling using the fitted StandardScaler from the training data
        # X = self.scaler.transform(X)
        
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
