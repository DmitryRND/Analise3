
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
import inspect

# Suppress warnings for a cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_model_default_params(model_constructor):
    """Extracts default parameters from a model's constructor."""
    try:
        sig = inspect.signature(model_constructor)
        return {
            p.name: p.default
            for p in sig.parameters.values()
            if p.default is not inspect.Parameter.empty
        }
    except Exception:
        return {}

# 1. Main training function - Rewritten for reliability
def train_model(model_name, train_series, forecast_horizon, future_covariates=None, model_params=None):
    """
    Constructs, trains a single Darts model, and returns the forecast and the trained model object.
    It now intelligently handles model-specific requirements and returns the model object for inspection.
    On failure, returns (None, None, error_message).
    """
    if model_params is None:
        model_params = {}

    model_info = MODELS[model_name]
    
    try:
        # --- INTELLIGENT PARAMETER INJECTION ---
        final_params = model_info['default_params'].copy()
        final_params.update(model_params) # User/Optuna params override defaults

        # Special handling for N-BEATS default parameters
        if model_name == "N-BEATS":
            # Ensure input_chunk_length is not > len(train_series)
            input_chunk = 2 * forecast_horizon
            if input_chunk >= len(train_series):
                input_chunk = max(1, len(train_series) - forecast_horizon -1)
                if input_chunk < 1:
                    return None, None, "Series too short for N-BEATS model."

            final_params.setdefault('input_chunk_length', input_chunk)
            final_params.setdefault('output_chunk_length', forecast_horizon)
            # N-BEATS is sensitive, remove unsupported params
            final_params.pop('lags', None)
            final_params.pop('lags_future_covariates', None)


        # For LightGBM, ensure output_chunk_length is set
        if model_name in ["LightGBM", "LinearRegression"]:
             final_params.setdefault('output_chunk_length', forecast_horizon)
             # These models also need lags
             final_params.setdefault('lags', forecast_horizon * 2)


        # Create a new model instance using the constructor
        model = model_info['constructor'](**final_params)

        # --- INTELLIGENT FITTING ---
        # Check if model supports future_covariates before passing them
        if model_info['requires_extras']:
            if future_covariates is None:
                return None, None, f"{model_name} requires external factors, but none were provided."
            if len(future_covariates) < len(train_series) + forecast_horizon:
                 return None, None, f"External factors for {model_name} are not long enough for the forecast horizon."
            
            model.fit(train_series, future_covariates=future_covariates)
            forecast = model.predict(forecast_horizon, future_covariates=future_covariates)
        else:
            # For models that don't support extras, don't pass them
            model.fit(train_series)
            forecast = model.predict(forecast_horizon)

        return forecast, model, None # Success!
    except Exception as e:
        return None, None, f"Error in {model_name}: {e}"


# 2. The MODELS dictionary with constructors and default parameters
MODELS = {
    "AutoARIMA": {
        "constructor": AutoARIMA,
        "requires_extras": False,
        "default_params": get_model_default_params(AutoARIMA),
    },
    "Prophet": {
        "constructor": Prophet,
        "requires_extras": False,
        "default_params": get_model_default_params(Prophet),
    },
    "ExponentialSmoothing": {
        "constructor": ExponentialSmoothing,
        "requires_extras": False,
        "default_params": get_model_default_params(ExponentialSmoothing),
    },
    "Theta": {
        "constructor": Theta,
        "requires_extras": False,
        "default_params": get_model_default_params(Theta),
    },
    # FFT is best used with covariates, so we mark it as requiring them
    "FFT": {
        "constructor": FFT,
        "requires_extras": True,
        "default_params": get_model_default_params(FFT),
    },
    "LightGBM": {
        "constructor": lambda **kwargs: LightGBMModel(random_state=42, **kwargs),
        "requires_extras": True,
        "default_params": get_model_default_params(LightGBMModel),
    },
    "LinearRegression": {
        "constructor": LinearRegressionModel,
        "requires_extras": True,
        "default_params": get_model_default_params(LinearRegressionModel),
    },
    "N-BEATS": {
        "constructor": lambda **kwargs: NBEATSModel(random_state=42, force_reset=True, pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}, **kwargs),
        "requires_extras": False,
        "default_params": get_model_default_params(NBEATSModel),
    },
}
