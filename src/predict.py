import pandas as pd
import wandb
import time

from _helpers import constants
from _helpers import functions as hf
from _helpers.verify_submission.verify_subm import main as verify_subm
from _helpers.score_submission.score_subm import main as score_subm

from models.model_random import ModelRandom
from models.model_baseline import ModelBaseline
from models.model_popular import ModelPopular
from models.model_log_reg import ModelLogisticRegression
from models.model_cheapest import ModelCheapest
from models.model_xgboost import ModelXGBoost
from models.model_mlp import ModelMLP
from models.model_neural import ModelNeural

# Initalize tool for logging
wandb.login()

# Provide model map
models = {
    'random': {
        'class': ModelRandom,
        'needs_data': True,
    },
    'baseline': {
        'class': ModelBaseline,
        'needs_data': True,
    },
    'popular': {
        'class': ModelPopular,
        'needs_data': True,
    },
    'cheapest': {
        'class': ModelCheapest
    },
    'log-reg': {
        'class': ModelLogisticRegression,
    },
    'xgboost': {
        'class': ModelXGBoost
    },
    'mlp': {
        'class': ModelMLP
    },
    'neural': {
        'class': ModelNeural
    }
}

# Tinker with the parameters
run = 1
notes = ''
params = {
    'model': 'neural',
    'timestamp': int(time.time()),
    'features': [
        "impressed_item_position",
        "relative_impressed_item_position",
        "impressed_item_rating",
        "user_impressed_item_interaction_count",
        "user_interacted_with_n_cheapest",
        "user_interacted_with_top_n",
        "price",
        "relative_price",
        "price_above_impression_mean",
        "is_last_interacted",
        "is_one_of_n_cheapest",
        "is_one_of_the_top_n",
    ],
    'epochs': 5,
    "batch_size": 64,
    'activation': 'softplus',
    'optimizer': 'adagrad',
    'neurons': (32, 16),
    'dropout': 0.1,
    'loss': 'mean_squared_error'
}

wandb_run = wandb.init(
    entity='mcfreddie777',
    project="dp-recsys",
    name=f'model_{params["model"]}_run_{run}',
    notes=notes
)
wandb_run.config.update(params)

# Choose model class
model = models[params['model']]['class']()
model.update(params)

# Load train data if needed for the model
df_train = []
if ('needs_data' in models[params['model']] and models[params['model']]['needs_data'] == True):
    df_train = pd.read_parquet(constants.DROPPED_TRAIN)

# Fit the model
model.fit(df_train, wandb=wandb_run)

# Load test data
df_test = pd.read_csv(constants.TEST)

# Predict
df_recommendations = model.predict(df_test)

# Verify predictions
verify_subm(df_subm=df_recommendations, df_test=df_test)

# Calculate submission score
df_gt = pd.read_csv(constants.GROUND_TRUTH)
mrr, map3 = score_subm(df_subm=df_recommendations, df_gt=df_gt)

wandb_run.log({"mrr": mrr, "map3": map3})
wandb_run.finish()
