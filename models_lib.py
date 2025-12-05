
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
    NHiTSModel,
    TCNModel,
)
# Пробуем импортировать CatBoostModel, если доступен
try:
    from darts.models import CatBoostModel
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
from darts.metrics import mae, mape, r2_score, rmse
import optuna
import warnings
import inspect
import numpy as np
import pandas as pd

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
                             future_covariates=None, n_trials=10, metric='mae', season_length=None):
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
    elif metric.lower() == 'rmse':
        metric_func = rmse
        direction = 'minimize'
    else:
        metric_func = mae
        direction = 'minimize'
    
    def objective(trial):
        try:
            # Определяем пространство гиперпараметров в зависимости от модели
            params = {}
            
            if model_name == "AutoARIMA":
                params['start_p'] = trial.suggest_int('start_p', 0, 2)
                params['start_q'] = trial.suggest_int('start_q', 0, 2)
                params['max_p'] = trial.suggest_int('max_p', 3, 8)
                params['max_q'] = trial.suggest_int('max_q', 3, 8)
                params['d'] = trial.suggest_int('d', 0, 2)
                params['max_d'] = trial.suggest_int('max_d', 1, 2)
                params['max_order'] = trial.suggest_int('max_order', 5, 12)
                params['stepwise'] = trial.suggest_categorical('stepwise', [True, False])
                params['stationary'] = trial.suggest_categorical('stationary', [True, False])
                if season_length and season_length > 1:
                    params['start_P'] = trial.suggest_int('start_P', 0, 1)
                    params['start_Q'] = trial.suggest_int('start_Q', 0, 1)
                    params['max_P'] = trial.suggest_int('max_P', 1, 4)
                    params['max_Q'] = trial.suggest_int('max_Q', 1, 4)
                    params['D'] = trial.suggest_int('D', 0, 1)
                    params['max_D'] = trial.suggest_int('max_D', 1, 2)
                    params['m'] = season_length
                    params['seasonal'] = True
                else:
                    params['seasonal'] = False

            elif model_name == "Prophet":
                params['yearly_seasonality'] = trial.suggest_categorical('yearly_seasonality', [True, False])
                params['weekly_seasonality'] = trial.suggest_categorical('weekly_seasonality', [True, False])
                params['daily_seasonality'] = trial.suggest_categorical('daily_seasonality', [True, False])
            
            elif model_name == "ExponentialSmoothing":
                # Для ExponentialSmoothing в darts параметры передаются напрямую
                # Используем только поддерживаемые значения
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
            forecast, _, error = train_model(
                model_name,
                train_series,
                forecast_horizon,
                future_covariates=future_covariates,
                model_params=params,
                season_length=season_length,
            )
            
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
def train_model(model_name, train_series, forecast_horizon, future_covariates=None, model_params=None, season_length=None):
    """
    Constructs, trains a single Darts model, and returns the forecast and the trained model object.
    It now intelligently handles model-specific requirements and returns the model object for inspection.
    On failure, returns (None, None, error_message).
    """
    def _ts_to_df(ts: TimeSeries):
        """Универсальное преобразование TimeSeries в DataFrame без пропусков."""
        try:
            if hasattr(ts, "pd_dataframe"):
                df = ts.pd_dataframe()
            elif hasattr(ts, "to_dataframe"):
                df = ts.to_dataframe()
            else:
                values = ts.values().flatten()
                df = pd.DataFrame({"value": values}, index=ts.time_index)
            df = df.dropna()
            return df
        except Exception:
            return None

    def _dropna_ts(ts: TimeSeries):
        df = _ts_to_df(ts)
        if df is None or df.empty:
            return ts
        df_reset = df.reset_index().rename(columns={df.index.name or "index": "time"})
        return TimeSeries.from_dataframe(
            df_reset,
            time_col="time",
            value_cols=[c for c in df_reset.columns if c != "time"],
            fill_missing_dates=False,
            freq=getattr(ts, "freq", None),
        )

    if model_params is None:
        model_params = {}

    model_info = MODELS[model_name]
    
    # Если целевая константная, некоторые модели (CatBoost) падают — пропускаем их
    if model_name == "CatBoost":
        df_target = _ts_to_df(train_series)
        if df_target is not None and not df_target.empty:
            y_vals = df_target.iloc[:, 0].to_numpy()
            if np.nanmax(y_vals) - np.nanmin(y_vals) == 0:
                return None, None, "CatBoost skipped: target is constant."

    try:
        # Жёстко убираем пропуски в target и ковариатах перед обучением
        if hasattr(train_series, "drop_missing_values"):
            train_series = train_series.drop_missing_values()
        else:
            train_series = _dropna_ts(train_series)
        if future_covariates is not None:
            if hasattr(future_covariates, "drop_missing_values"):
                future_covariates = future_covariates.drop_missing_values()
            else:
                future_covariates = _dropna_ts(future_covariates)

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


        # Inject seasonality into models that support it (attempt, fallback-safe, only if param exists)
        if model_name == "AutoARIMA":
            if season_length and season_length > 1:
                final_params["m"] = season_length
                final_params["seasonal"] = True
            else:
                final_params["seasonal"] = False
        if model_name == "ExponentialSmoothing" and season_length:
            final_params.setdefault('seasonal_periods', season_length)
        if model_name == "N-HiTS":
            # Гибкая длина окна: минимум 2, максимум весь ряд минус 1
            max_input = max(2, len(train_series) - 1)
            input_chunk = max(2, min(max_input, forecast_horizon * 2))
            final_params.setdefault('input_chunk_length', input_chunk)
            final_params.setdefault('output_chunk_length', forecast_horizon)
            final_params.setdefault('random_state', 42)
        if model_name == "TCN":
            max_input = max(2, len(train_series) - 1)
            input_chunk = max(2, min(max_input, forecast_horizon * 2))
            final_params.setdefault('input_chunk_length', input_chunk)
            final_params.setdefault('output_chunk_length', forecast_horizon)
            final_params.setdefault('kernel_size', 3)
            final_params.setdefault('num_filters', 5)
            final_params.setdefault('dilation_base', 2)
            final_params.setdefault('dropout', 0.1)
            final_params.setdefault('weight_norm', True)
            final_params.setdefault('random_state', 42)
        if model_name == "Theta" and season_length:
            # Darts Theta uses seasonality_period
            final_params.setdefault('seasonality_period', season_length)

        # For LightGBM and LinearRegression, ensure output_chunk_length is set
        if model_name in ["LightGBM", "LinearRegression", "CatBoost"]:
            final_params.setdefault('output_chunk_length', forecast_horizon)
            
            # Set lags for the target variable
            if 'lags' not in final_params or final_params['lags'] is None:
                # Use a reasonable default for lags based on the forecast horizon
                lags = min(forecast_horizon * 2, len(train_series) - forecast_horizon - 1)
                final_params['lags'] = lags if lags > 0 else 1
            
            # Handle future covariates if they exist
            if future_covariates is not None:
                # Ensure lags_future_covariates is properly set
                if 'lags_future_covariates' not in final_params or final_params['lags_future_covariates'] is None:
                    # Use the same lags as the target variable by default
                    lag_val = final_params.get('lags', forecast_horizon)
                    # Ensure we don't exceed the available data
                    max_lag = min(lag_val, len(train_series) - forecast_horizon - 1)
                    if max_lag > 0:
                        final_params['lags_future_covariates'] = list(range(1, max_lag + 1))
                    else:
                        # If we can't have any lags, use a minimal lag of 1
                        final_params['lags_future_covariates'] = [1]
                
                # Ensure lags_future_covariates is a list or tuple
                if isinstance(final_params['lags_future_covariates'], int):
                    final_params['lags_future_covariates'] = [final_params['lags_future_covariates']]
                elif not isinstance(final_params['lags_future_covariates'], (list, tuple)):
                    final_params['lags_future_covariates'] = [1]  # Default to lag 1 if invalid
        
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
                    "additive": ModelMode.ADDITIVE,         # добавили
                    "mul": ModelMode.MULTIPLICATIVE,
                    "multiplicative": ModelMode.MULTIPLICATIVE, # добавили
                    "none": ModelMode.NONE,
                    "n": ModelMode.NONE,
                    "": None
                }
                final_params['trend'] = trend_map.get(tr.lower(), None)

            seas = final_params.get('seasonal')
            if isinstance(seas, str):
                seasonal_map = {
                    "add": SeasonalityMode.ADDITIVE,
                    "additive": SeasonalityMode.ADDITIVE,         # добавили
                    "mul": SeasonalityMode.MULTIPLICATIVE,
                    "multiplicative": SeasonalityMode.MULTIPLICATIVE, # добавили
                    "none": SeasonalityMode.NONE,
                    "n": SeasonalityMode.NONE,
                    "": None
                }
                final_params['seasonal'] = seasonal_map.get(seas.lower(), None)
    
        # Create a new model instance using the constructor
        if model_name == "AutoARIMA" and season_length and season_length > 1:
            final_params.pop("min_d", None)  # может не поддерживаться
            seasonal_variants = [
                {"m": season_length, "seasonal": True},
            ]
        else:
            seasonal_variants = [{}]

        model = None
        last_error = None
        for variant in seasonal_variants:
            params_variant = final_params.copy()
            params_variant.update(variant)
            try:
                model = model_info['constructor'](**params_variant)
                break
            except TypeError as e:
                last_error = e
                continue

        if model is None:
            # Final fallback without seasonal keys
            cleanup_keys = ["m", "season_length", "seasonal_periods", "seasonal"]
            cleaned_params = {k: v for k, v in final_params.items() if k not in cleanup_keys}
            try:
                model = model_info['constructor'](**cleaned_params)
            except Exception:
                if last_error:
                    raise last_error
                else:
                    raise

        # --- INTELLIGENT FITTING ---
        # Drop NaNs that can break sklearn-based models
        try:
            ts_df = _ts_to_df(train_series)
            if ts_df is not None and ts_df.isna().values.any():
                train_series = _dropna_ts(train_series)
        except Exception:
            pass
        if future_covariates is not None:
            try:
                fc_df = _ts_to_df(future_covariates)
                if fc_df is not None and fc_df.isna().values.any():
                    future_covariates = _dropna_ts(future_covariates)
            except Exception:
                pass

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
    "N-HiTS": {
        "constructor": lambda **kwargs: NHiTSModel(
            force_reset=True,
            pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False},
            **kwargs,
        ),
        "requires_extras": False,
        "default_params": get_model_default_params(NHiTSModel),
    },
    "TCN": {
        "constructor": lambda **kwargs: TCNModel(**kwargs),
        "requires_extras": False,
        "default_params": get_model_default_params(TCNModel),
    },
}

# Добавляем CatBoost, если доступен
if CATBOOST_AVAILABLE:
    MODELS["CatBoost"] = {
        "constructor": lambda **kwargs: CatBoostModel(random_state=42, verbose=False, **kwargs),
        "requires_extras": False,  # Может работать без экзогенных переменных
        "default_params": get_model_default_params(CatBoostModel),
    }
