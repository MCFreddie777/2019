import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input, Dropout
import tensorflow as tf
import joblib
import copy
import wandb

from _helpers import constants
from _helpers import functions as hf


class ModelNeural():
    """
    Model class using neural network
    """
    params = {}
    
    def __get_features_and_labels(self, filename):
        if type(filename) is list or type(filename) is tuple:
            df = hf.concat_files(filename)
        else:
            df = pd.read_parquet(filename)
        
        df.loc[:, "is_clicked"] = (
                df["referenced_item"] == df["impressed_item"]
        ).astype(int)
        
        X = df[self.params['features']]
        y = df['is_clicked']
        
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
            neurons=self.params['neurons'],
            activation=self.params['activation'],
            optimizer=self.params['optimizer'],
            loss=self.params['loss'],
            dropout=self.params['dropout']
        )
        
        # Split the data into training and validation sets
        train_chunks = hf.get_preprocessed_dataset_chunks('train')
        train_chunks, val_chunks = train_test_split(train_chunks, test_size=.20, shuffle=True)
        
        # TODO: Scaling
        # features_to_scale = [f for f in ['price'] if f in self.params['features']]
        # self.scaler = MinMaxScaler()
        
        callbacks = tf.keras.callbacks.CallbackList(
            None,
            add_history=True,
            add_progbar=True,
            model=self.model,
            epochs=self.params['epochs'],
            verbose=1,
            steps=len(train_chunks)
        )
        training_logs = None
        
        # Train in epochs
        callbacks.on_train_begin()
        
        for epoch in range(self.params['epochs']):
            self.model.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            
            logs = None
            
            # Partially fit the best estimator on subsequent chunks
            for i, chunk_filename in enumerate(train_chunks):
                callbacks.on_train_batch_begin(i)
                
                X_train, y_train = self.__get_features_and_labels(chunk_filename)
                
                # Scale train data
                # X_train[[f'{col}_scaled' for col in features_to_scale]] = \
                #     self.scaler.partial_fit(X_train[features_to_scale].values)
                
                logs = self.model.train_on_batch(
                    x=X_train,
                    y=y_train,
                    reset_metrics=False,
                    return_dict=True,
                )
                
                callbacks.on_train_batch_end(i, logs)
                wandb.log(logs)
            
            epoch_logs = copy.copy(logs)
            
            # Validation at the end of the epoch
            # X_val, y_val = self.__get_features_and_labels(val_chunks)
            
            # Perform feature scaling using the fitted scaler on the validation data
            # X_val[[f'{col}_scaled' for col in features_to_scale]] = self.scaler.transform(X_val[features_to_scale])
            
            # validation_logs = self.model.evaluate(X_val, y_val, callbacks=callbacks, return_dict=True)
            # epoch_logs.update({'val_' + name: v for name, v in validation_logs.items()})
            
            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            
            wandb.log(epoch_logs)
            wandb.log({'epoch': epoch})
        
        callbacks.on_train_end(logs=training_logs)
        wandb.log(training_logs)
        
        # Persist model in file
        joblib.dump(self.model, constants.MODEL_DIR / f'{self.params["model"]}_{self.params["timestamp"]}')
    
    def predict(self, *args, **kwargs):
        df_impressions = hf.load_preprocessed_dataset('test')
        
        df_impressions = df_impressions[df_impressions.referenced_item.isna()]
        
        X = df_impressions[self.params['features']]
        
        # Perform feature scaling using the fitted scaler from the training data
        # features_to_scale = [f for f in ['price'] if f in self.params['features']]
        # X[[f'{col}_scaled' for col in features_to_scale]] = self.scaler.transform(X[features_to_scale])
        
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
