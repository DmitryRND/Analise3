
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
# Пробуем импортировать CatBoostModel, если доступен
try:
    from darts.models import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
from darts.metrics import mae, mape, r2_score
import optuna
import warnings
import inspect
import numpy as np
from darts.utils.utils import ModelMode, SeasonalityMode

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


def optimize_hyperparameters(model_name, train_series, val_series, forecast_horizon, 
                             future_covariates=None, n_trials=10, metric='mae'):
    """
    Оптимизирует гиперпараметры модели с помощью Optuna.
    Возвращает лучшие параметры или None при ошибке.
    """
    model_info = MODELS.get(model_name)
    if not model_info:
        return None, f"Модель {model_name} не найдена"
    
    # Определяем метрику для оптимизации
    if metric.lower() == 'mae':
        metric_func = mae
        direction = 'minimize'
    elif metric.lower() == 'mape':
        metric_func = mape
        direction = 'minimize'
    elif metric.lower() == 'mse':
        from darts.metrics import mse
        metric_func = mse
        direction = 'minimize'
    elif metric.lower() == 'r2':
        metric_func = r2_score
        direction = 'maximize'
    else:
        metric_func = mae
        direction = 'minimize'
    
    def objective(trial):
        try:
            # Определяем пространство гиперпараметров в зависимости от модели
            params = {}
            
            if model_name == "Prophet":
                params['yearly_seasonality'] = trial.suggest_categorical('yearly_seasonality', [True, False])
                params['weekly_seasonality'] = trial.suggest_categorical('weekly_seasonality', [True, False])
                params['daily_seasonality'] = trial.suggest_categorical('daily_seasonality', [True, False])
            
            elif model_name == "ExponentialSmoothing":
                params['trend'] = trial.suggest_categorical('trend', [ModelMode.ADDITIVE, ModelMode.MULTIPLICATIVE, ModelMode.NONE, None])
                params['seasonal'] = trial.suggest_categorical('seasonal', [SeasonalityMode.ADDITIVE, SeasonalityMode.MULTIPLICATIVE, SeasonalityMode.NONE, None])
                params['seasonal_periods'] = trial.suggest_int('seasonal_periods', 2, 24)
            
            elif model_name == "LightGBM":
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                params['lags'] = trial.suggest_int('lags', forecast_horizon, min(forecast_horizon * 4, len(train_series) - forecast_horizon))
                if future_covariates is not None:
                    lag_val = trial.suggest_int('lags_future_covariates', 1, min(forecast_horizon, len(train_series) - forecast_horizon))
                    params['lags_future_covariates'] = tuple(range(1, lag_val + 1))
            
            elif model_name == "LinearRegression":
                params['lags'] = trial.suggest_int('lags', forecast_horizon, min(forecast_horizon * 4, len(train_series) - forecast_horizon))
                if future_covariates is not None:
                    lag_val = trial.suggest_int('lags_future_covariates', 1, min(forecast_horizon, len(train_series) - forecast_horizon))
                    params['lags_future_covariates'] = tuple(range(1, lag_val + 1))
            
            elif model_name == "CatBoost":
                params['iterations'] = trial.suggest_int('iterations', 50, 500)
                params['depth'] = trial.suggest_int('depth', 3, 10)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                params['lags'] = trial.suggest_int('lags', forecast_horizon, min(forecast_horizon * 4, len(train_series) - forecast_horizon))
                if future_covariates is not None:
                    lag_val = trial.suggest_int('lags_future_covariates', 1, min(forecast_horizon, len(train_series) - forecast_horizon))
                    params['lags_future_covariates'] = tuple(range(1, lag_val + 1))
            
            elif model_name == "N-BEATS":
                input_chunk = trial.suggest_int('input_chunk_length', max(1, forecast_horizon), min(len(train_series) - forecast_horizon - 1, forecast_horizon * 3))
                if input_chunk < 1:
                    input_chunk = max(1, len(train_series) - forecast_horizon - 1)
                params['input_chunk_length'] = input_chunk
                params['output_chunk_length'] = forecast_horizon
                params['n_epochs'] = trial.suggest_int('n_epochs', 10, 50)
            
            # Обучаем модель с этими параметрами
            forecast, _, error = train_model(model_name, train_series, forecast_horizon, 
                                            future_covariates=future_covariates, model_params=params)
            
            if error or forecast is None:
                return float('inf') if direction == 'minimize' else float('-inf')
            
            # Вычисляем метрику на валидационной выборке
            score = metric_func(val_series, forecast)
            
            # Обрабатываем NaN и Inf
            if np.isnan(score) or np.isinf(score):
                return float('inf') if direction == 'minimize' else float('-inf')
            
            return score if direction == 'minimize' else -score  # Для maximize инвертируем
            
        except Exception as e:
            return float('inf') if direction == 'minimize' else float('-inf')
    
    try:
        study = optuna.create_study(direction=direction, study_name=f"{model_name}_optimization")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        if len(study.trials) == 0 or study.best_trial is None:
            return None, "Не удалось найти оптимальные параметры"
        
        best_params = study.best_trial.params.copy()
        
        return best_params, None
    except Exception as e:
        return None, f"Ошибка оптимизации: {e}"

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


        # For LightGBM and LinearRegression, ensure output_chunk_length is set
        if model_name in ["LightGBM", "LinearRegression", "CatBoost"]:
             final_params.setdefault('output_chunk_length', forecast_horizon)
             # These models also need lags (but can work without future_covariates)
             if 'lags' not in final_params or final_params['lags'] is None:
                 final_params['lags'] = min(forecast_horizon * 2, len(train_series) - forecast_horizon)
             # Если используются future_covariates, нужно установить lags_future_covariates (должен быть tuple или list)
             if future_covariates is not None:
                 if 'lags_future_covariates' not in final_params or final_params['lags_future_covariates'] is None:
                     lag_val = min(forecast_horizon, len(train_series) - forecast_horizon)
                     final_params['lags_future_covariates'] = tuple(range(1, lag_val + 1)) if lag_val > 0 else (1,)
        
        # Для LightGBM и CatBoost добавляем параметры, чтобы убрать предупреждения и ускорить работу
        if model_name == "LightGBM":
             # Увеличиваем минимальное количество данных в листе для стабильности
             if len(train_series) < 100:
                 min_leaf = max(1, len(train_series) // 20)
             else:
                 min_leaf = max(5, len(train_series) // 50)
             final_params.setdefault('min_data_in_leaf', min_leaf)
             final_params.setdefault('min_sum_hessian_in_leaf', 1e-3)
             # Отключаем verbose для ускорения
             final_params.setdefault('verbose', -1)
             # Ограничиваем количество деревьев для ускорения, если не задано явно
             if 'n_estimators' not in final_params and 'num_iterations' not in final_params:
                 final_params.setdefault('n_estimators', min(100, len(train_series) // 10))
        
        if model_name == "CatBoost":
             # Не устанавливаем verbose здесь, так как он уже установлен в constructor
             # Ограничиваем количество итераций для ускорения
             if 'iterations' not in final_params:
                 final_params.setdefault('iterations', min(100, len(train_series) // 10))
             # Удаляем verbose из final_params, так как он уже в constructor
             final_params.pop('verbose', None)
        
        # Для Prophet улучшаем сезонность на основе длины данных
        if model_name == "Prophet":
            try:
                # Получаем значения для определения режима сезонности
                from utils import _get_ts_values_and_index
                ts_values, _ = _get_ts_values_and_index(train_series)
                
                # Автоматически определяем сезонность на основе длины ряда
                # Если данных больше 730 (примерно 2 года дневных данных) - включаем годовую сезонность
                if len(train_series) > 730:
                    final_params.setdefault('yearly_seasonality', True)
                
                # Если данных больше 60 (примерно 2 месяца дневных данных) - включаем недельную сезонность
                if len(train_series) > 60:
                    final_params.setdefault('weekly_seasonality', True)
                
                # Дневная сезонность по умолчанию отключена для месячных данных
                final_params.setdefault('daily_seasonality', False)
                
                # Определяем режим сезонности: если все значения положительны, используем multiplicative
                if np.min(ts_values) > 0:
                    final_params.setdefault('seasonality_mode', 'multiplicative')
                else:
                    final_params.setdefault('seasonality_mode', 'additive')
            except:
                # В случае ошибки используем значения по умолчанию
                pass
        
        if model_name == "ExponentialSmoothing":
            tr = final_params.get('trend')
            if isinstance(tr, str):
                trend_map = {
                    "add": ModelMode.ADDITIVE,
                    "additive": ModelMode.ADDITIVE,
                    "mul": ModelMode.MULTIPLICATIVE,
                    "multiplicative": ModelMode.MULTIPLICATIVE,
                    "none": ModelMode.NONE,
                    "n": ModelMode.NONE,
                    "": None
                }
                final_params['trend'] = trend_map.get(tr.lower(), None)

            seas = final_params.get('seasonal')
            if isinstance(seas, str):
                seasonal_map = {
                    "add": SeasonalityMode.ADDITIVE,
                    "additive": SeasonalityMode.ADDITIVE,
                    "mul": SeasonalityMode.MULTIPLICATIVE,
                    "multiplicative": SeasonalityMode.MULTIPLICATIVE,
                    "none": SeasonalityMode.NONE,
                    "n": SeasonalityMode.NONE,
                    "": None
                }
                final_params['seasonal'] = seasonal_map.get(seas.lower(), None)

        # Create a new model instance using the constructor
        model = model_info['constructor'](**final_params)

        # --- INTELLIGENT FITTING ---
        # FFT не поддерживает future_covariates вообще
        if model_name == "FFT":
            # FFT работает без экзогенных переменных в darts
            model.fit(train_series)
            forecast = model.predict(forecast_horizon)
        # LightGBM, LinearRegression и CatBoost can work with or without future_covariates
        elif model_name in ["LightGBM", "LinearRegression", "CatBoost"]:
            if future_covariates is not None:
                if len(future_covariates) < len(train_series) + forecast_horizon:
                    return None, None, f"External factors for {model_name} are not long enough for the forecast horizon."
                model.fit(train_series, future_covariates=future_covariates)
                forecast = model.predict(forecast_horizon, future_covariates=future_covariates)
            else:
                # Work without future_covariates
                model.fit(train_series)
                forecast = model.predict(forecast_horizon)
        elif model_info['requires_extras']:
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
    # FFT не поддерживает экзогенные переменные в darts
    "FFT": {
        "constructor": FFT,
        "requires_extras": False,  # FFT работает без экзогенных переменных
        "default_params": get_model_default_params(FFT),
    },
    "LightGBM": {
        "constructor": lambda **kwargs: LightGBMModel(random_state=42, **kwargs),
        "requires_extras": False,  # Может работать без экзогенных переменных
        "default_params": get_model_default_params(LightGBMModel),
    },
    "LinearRegression": {
        "constructor": LinearRegressionModel,
        "requires_extras": False,  # Может работать без экзогенных переменных
        "default_params": get_model_default_params(LinearRegressionModel),
    },
    "N-BEATS": {
        "constructor": lambda **kwargs: NBEATSModel(random_state=42, force_reset=True, pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}, **kwargs),
        "requires_extras": False,
        "default_params": get_model_default_params(NBEATSModel),
    },
}

# Добавляем CatBoost, если доступен
if CATBOOST_AVAILABLE:
    MODELS["CatBoost"] = {
        "constructor": lambda **kwargs: CatBoostModel(random_state=42, verbose=False, **kwargs),
        "requires_extras": False,  # Может работать без экзогенных переменных
        "default_params": get_model_default_params(CatBoostModel),
    }
