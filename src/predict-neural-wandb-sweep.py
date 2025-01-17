import pandas as pd
import numpy as np
import wandb
import time
from functools import partial
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Dropout
import joblib
from wandb.keras import WandbCallback

from _helpers.verify_submission.verify_subm import main as verify_subm
from _helpers.score_submission.score_subm import main as score_subm

from _helpers import constants
from _helpers import functions as hf


class ModelNeural():
    """
    Model class using neural network
    """
    params = {}
    
    def __train_test_split(self, df, test_size=0.2):
        """
        Splits dataset into train and validation with keeping session data integrity
        """
        unique_sessions = df.loc[df['session_id'].duplicated(keep=False), 'session_id'].unique()
        np.random.shuffle(unique_sessions)
        
        split_idx = int(len(unique_sessions) * (1 - test_size))  # 80% for train
        
        train_df = df[df['session_id'].isin(unique_sessions[:split_idx])]
        test_df = df[df['session_id'].isin(unique_sessions[split_idx:])]
        
        return train_df, test_df
    
    def __scale_features(self, X, features_to_scale):
        features = X[features_to_scale]
        
        scaler = StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        
        X[features_to_scale] = features
        
        return scaler, X
    
    def __get_features_and_labels(self, filename, val_size):
        df = pd.read_parquet(filename)
        
        df.loc[:, "is_clicked"] = (
                df["referenced_item"] == df["impressed_item"]
        ).astype(int)
        
        if val_size is not None:
            df_train, df_val = self.__train_test_split(df, test_size=val_size)
            X_mask = self.params['features']
            y_mask = ["is_clicked"]
            return df_train[X_mask], df_val[X_mask], df_train[y_mask], df_val[y_mask]
        
        else:
            X = df[self.params['features']]
            y = df.is_clicked
            return X, y
    
    def __create_model(
            self,
            input_shape,
            activation='relu',
            optimizer='adam',
            neurons=(64, 32),
            loss='binary_crossentropy',
            dropout=0.1
    ):
        model = Sequential()
        
        model.add(Input(shape=input_shape))
        model.add(SimpleRNN(units=neurons[0], activation=activation))
        model.add(Dense(units=neurons[1], activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss=loss, optimizer=optimizer)
        
        return model
    
    def update(self, params):
        self.params = params
    
    def fit(self, *args, **kwargs):
        # Create the neural network model
        self.model = self.__create_model(
            input_shape=(len(self.params['features']), 1),
            neurons=wandb.config.neurons,
            activation=wandb.config.activation,
            optimizer=wandb.config.optimizer,
            loss=wandb.config.loss,
            dropout=wandb.config.dropout
        )
        
        # Split the data into training and validation sets
        train_chunks = hf.get_preprocessed_dataset_chunks('train')
        
        # Train in epochs
        for epoch in range(wandb.config.epochs):
            # Partially fit the best estimator on subsequent chunks
            for i, chunk_filename in enumerate(train_chunks):
                X_train, X_val, y_train, y_val = self.__get_features_and_labels(chunk_filename, val_size=0.2)
                
                self.scaler, X = self.__scale_features(
                    X_train,
                    [f for f in ['impressed_item_position', 'impressed_item_rating', 'price'] if
                     f in self.params['features']]
                )
                self.model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=1,
                    batch_size=wandb.config.batch_size,
                    callbacks=[WandbCallback()],
                )
            
            # Persist model in file
            joblib.dump(self.model, constants.MODEL_DIR / f'{self.params["model"]}_{self.params["timestamp"]}')
    
    def predict(self, *args, **kwargs):
        df_impressions = hf.load_preprocessed_dataset('test')
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        X = df_impressions[self.params['features']]
        
        # Perform feature scaling using the fitted StandardScaler from the training data
        features_to_scale = [f for f in ['impressed_item_position', 'impressed_item_rating', 'price'] if
                             f in self.params['features']]
        X[features_to_scale] = self.scaler.transform(X[features_to_scale])
        
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


def train(df_test, df_gt, run, notes, params):
    wandb.init(
        entity='mcfreddie777',
        project="dp-recsys",
        name=f'model_{params["model"]}_run_{run}',
        notes=notes
    )
    wandb.config.update(params)
    
    model = ModelNeural()
    model.update(params)
    
    # Train the model
    model.fit()
    
    # Predict
    df_recommendations = model.predict()
    
    # Verify predictions
    verify_subm(df_subm=df_recommendations, df_test=df_test)
    
    # Calculate submission score
    mrr, map3 = score_subm(df_subm=df_recommendations, df_gt=df_gt)
    
    wandb.log({"mrr": mrr, "map3": map3})


def main():
    # Initalize tool for logging
    wandb.login()
    
    # Tinker with the parameters
    run = 10
    notes = 'Optimalize hyperparameters with WandB sweeps'
    params = {
        'model': 'neural',
        'timestamp': int(time.time()),
        'features': [
            "impressed_item_position",
            "impressed_item_rating",
            "user_impressed_item_interaction_count",
            "price",
            "price_above_impression_mean",
            "is_last_interacted",
        ],
    }
    
    sweep_config = {
        'method': 'random',  # grid, random
        'metric': {
            'name': 'mrr',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {
                'values': [2, 5, 10]
            },
            'batch_size': {
                'values': [64, 32]
            },
            'optimizer': {
                'values': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
            },
            'activation': {
                'values': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
            },
            'dropout': {
                'values': [0, 1e-3, 1e-2, 0.1, 0.3, 0.5]
            },
            'neurons': {
                'values': [(64, 32), (32, 16), (8, 4)],
            },
            'loss': {
                'values': ['binary_crossentropy', 'mean_squared_error']
            },
        },
    }
    
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity='mcfreddie777',
        project="dp-recsys",
    )
    
    # Load test data
    df_test = pd.read_csv(constants.TEST)
    
    # Load ground truth
    df_gt = pd.read_csv(constants.GROUND_TRUTH)
    
    # Fit the model
    wandb.agent(
        sweep_id=sweep_id,
        entity='mcfreddie777',
        project="dp-recsys",
        function=partial(train, df_test=df_test, df_gt=df_gt, run=run, notes=notes, params=params)
    )


if __name__ == "__main__":
    main()
