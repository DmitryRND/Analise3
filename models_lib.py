
import pandas as pd
from darts import TimeSeries
from darts.models import (
    Prophet,
    AutoARIMA,
    ExponentialSmoothing,
    Theta,
    FFT,
    LightGBMModel,
    LinearRegressionModel,
    NBEATSModel,
)
from darts.metrics import mae, mape, r2_score
import optuna
import warnings

# Suppress Optuna's trial pruning messages
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)


# 1. Main training function
def train_model(model_class, train_series, forecast_horizon, future_covariates=None, model_params=None):
    """
    Trains a single Darts model and returns the forecast.
    """
    if model_params is None:
        model_params = {}
    try:
        # Handle models that require specific chunk lengths
        if 'input_chunk_length' in model_class.constructors:
             if 'input_chunk_length' not in model_params:
                model_params['input_chunk_length'] = max(30, forecast_horizon * 2)
             if 'output_chunk_length' not in model_params:
                model_params['output_chunk_length'] = forecast_horizon

        model = model_class(**model_params)
        model.fit(train_series, future_covariates=future_covariates)
        forecast = model.predict(forecast_horizon, future_covariates=future_covariates)
        return forecast, None
    except Exception as e:
        return None, str(e)

# 2. Hyperparameter optimization function
def optimize_hyperparameters(model_name, train_series, val_series, forecast_horizon, future_covariates=None, n_trials=20):
    """
    Optimizes hyperparameters for a given model using Optuna.
    """
    study_function = OPTIMIZATION_MAP.get(model_name)
    if not study_function:
        return None, "Optimization not implemented for this model."

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: study_function(trial, train_series, val_series, forecast_horizon, future_covariates), n_trials=n_trials)

        best_params = study.best_params
        model_info = MODELS[model_name]
        model_class = model_info['model']

        # Re-create model with best params and retrain on full data
        full_train_series = train_series.append(val_series)
        
        forecast, error = train_model(
            model_class, 
            full_train_series, 
            forecast_horizon, 
            future_covariates=future_covariates, 
            model_params=best_params
        )
        
        if error:
            raise Exception(error)
            
        return forecast, None

    except Exception as e:
        return None, str(e)


# 3. Helper functions for each model's optimization study
def _study_prophet(trial, train_series, val_series, forecast_horizon, future_covariates):
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
    }
    model = Prophet(**params)
    model.fit(train_series)
    forecast = model.predict(len(val_series))
    return mae(val_series, forecast)

def _study_lgbm(trial, train_series, val_series, forecast_horizon, future_covariates):
    params = {
        "lags": trial.suggest_int("lags", 1, 30),
        "lags_future_covariates": trial.suggest_int("lags_future_covariates", 0, 15) if future_covariates else 0,
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
    }
    model = LightGBMModel(**params, random_state=42)
    model.fit(train_series, future_covariates=future_covariates)
    forecast = model.predict(len(val_series), future_covariates=future_covariates)
    return mae(val_series, forecast)

def _study_nbeats(trial, train_series, val_series, forecast_horizon, future_covariates):
    input_chunk_length = trial.suggest_int("input_chunk_length", 10, 50)
    params = {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": trial.suggest_int("output_chunk_length", 1, min(input_chunk_length-1, 20)),
        "num_stacks": trial.suggest_int("num_stacks", 2, 10),
        "num_blocks": trial.suggest_int("num_blocks", 1, 5),
        "num_layers": trial.suggest_int("num_layers", 2, 8),
        "layer_widths": trial.suggest_int("layer_widths", 128, 1024),
        "n_epochs": trial.suggest_int("n_epochs", 20, 100),
    }
    model = NBEATSModel(**params, random_state=42)
    model.fit(train_series, verbose=False)
    forecast = model.predict(len(val_series))
    return mae(val_series, forecast)


# 4. The MODELS dictionary
MODELS = {
    "AutoARIMA": {
        "model": AutoARIMA,
        "requires_extras": False,
        "optimization_implemented": False,
    },
    "Prophet": {
        "model": Prophet,
        "requires_extras": False,
        "optimization_implemented": True,
    },
    "ExponentialSmoothing": {
        "model": ExponentialSmoothing,
        "requires_extras": False,
        "optimization_implemented": False,
    },
    "Theta": {
        "model": Theta,
        "requires_extras": False,
        "optimization_implemented": False,
    },
    "FFT": {
        "model": FFT,
        "requires_extras": False,
        "optimization_implemented": False,
    },
    "LightGBM": {
        "model": LightGBMModel,
        "requires_extras": True,
        "optimization_implemented": True,
    },
    "LinearRegression": {
        "model": LinearRegressionModel,
        "requires_extras": True,
        "optimization_implemented": False,
    },
    "N-BEATS": {
        "model": NBEATSModel,
        "requires_extras": False,
        "optimization_implemented": True,
    },
}

# 5. Mapping for optimization functions
OPTIMIZATION_MAP = {
    "Prophet": _study_prophet,
    "LightGBM": _study_lgbm,
    "N-BEATS": _study_nbeats,
}
