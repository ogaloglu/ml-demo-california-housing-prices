"""Training script for XGBoost model using WandB integration for logging."""
import logging
import math

import wandb
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.utils import get_data, load_file, save_file

logger = logging.getLogger(__name__)

X_train, X_val, y_train, y_val, *_ = get_data()
logger.info("Dataset is obtained.")

preprocessor = load_file("preprocessor.joblib")
# Preprocessor is already fit to X_train before.
preprocessed_X_train = preprocessor.transform(X_train)
preprocessed_X_val = preprocessor.transform(X_val)
logger.info("Dataset is preprocessed.")

best_r2 = 0


def train_model():
    global best_r2
    wandb.init()

    model = XGBRegressor(
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate,
        n_estimators=wandb.config.n_estimators,
        colsample_bytree=wandb.config.colsample_bytree,
        min_child_weight=wandb.config.min_child_weight,
    )
    model.fit(preprocessed_X_train, y_train)

    # Predict on test set
    y_preds = model.predict(preprocessed_X_val)

    # Evaluate predictions
    r2_result = r2_score(y_val, y_preds)
    rmse_result = math.sqrt(mean_squared_error(y_val, y_preds))

    # Save model if R2 score is better than the previous best model
    if r2_result > best_r2:
        logger.info("Current model scores the best.")
        best_r2 = r2_result
        save_file(model, "best_model.joblib")
        logger.info("Current model is saved.")

    # Log model performance metrics to W&B
    wandb.log({"r2": r2_result, "rmse": rmse_result})


sweep_configs = {
    "method": "bayes",
    "metric": {"name": "mse", "goal": "minimize"},
    "parameters": {
        "colsample_bytree": {"distribution": "uniform", "min": 0.5, "max": 1},
        "max_depth": {"values": [2, 5, 10]},
        "learning_rate": {"distribution": "uniform", "min": 0.05, "max": 0.15},
        "min_child_weight": {"distribution": "uniform", "min": 0, "max": 10},
        "n_estimators": {"values": [100, 500, 1000, 2000]},
    },
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_configs, project="california-housing-prices")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=30)
    logger.info(f"R2 score of the best model: {best_r2}")
