from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier

from _helpers import functions as hf


class ModelNeural():
    """
    Model class using neural network
    """
    params = {}
    
    def __create_model(input_shape, activation='relu', optimizer='adam', neurons=(64, 32)):
        model = Sequential()
        
        model.add(Dense(neurons[0], input_shape, activation=activation))
        model.add(Dense(neurons[1], activation=activation))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        return model
    
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
        
        # Define the parameter grid
        param_grid = {
            'epochs': [10, 20],
            'batch_size': [32, 64],
            'neurons': [(128, 64), (64, 32), (32, 16)],
            'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
            'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
            'learn_rate': [.001, 0.01, 0.1, 0.2, 0.3],
            'momentum': [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        }
        
        # Create the neural network model
        model = KerasClassifier(model=self.__create_model, input_shape=(X.shape[1],), verbose=True)
        
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
