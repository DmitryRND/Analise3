import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from io import BytesIO
from darts.utils.utils import ModelMode, SeasonalityMode
import plotly.colors

def map_model_mode(val_str):
    """Превращает строку в Enum для Darts (Trend/Seasonality)"""
    if val_str == "additive": return ModelMode.ADDITIVE
    if val_str == "multiplicative": return ModelMode.MULTIPLICATIVE
    return ModelMode.NONE


def map_seasonality_mode(val_str):
    """Превращает строку в Enum для Darts (Theta/Prophet)"""
    if val_str == "additive": return SeasonalityMode.ADDITIVE
    if val_str == "multiplicative": return SeasonalityMode.MULTIPLICATIVE
    return SeasonalityMode.ADDITIVE


def get_safe_lags(data_len):
    """Вычисляет безопасное количество лагов для ML моделей"""
    max_lags = int(data_len / 2) - 1
    return min(12, max_lags) if max_lags > 1 else 1


def detect_outliers_zscore(df, value_col):
    """Определяет есть ли выбросы (Z-score > 3)"""
    if len(df) < 10: return False
    numeric_series = pd.to_numeric(df[value_col], errors='coerce').dropna()
    z = np.abs(stats.zscore(numeric_series))
    return np.any(z > 3)


def plot_decomposition(df, value_col, period):
    """Строит график декомпозиции (Тренд, Сезонность, Шум)"""
    try:
        if len(df) < period * 2: return None

        decomposition = seasonal_decompose(df[value_col], model='additive', period=int(period))

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Тренд (Направление)", "Сезонность (Циклы)", "Остатки (Шум)"),
                            vertical_spacing=0.1)

        fig.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend', line=dict(color='#FF9900')),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal',
                                 line=dict(color='#00CC66')), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Resid', mode='markers',
                                 marker=dict(color='gray', size=3)), row=3, col=1)

        fig.update_layout(height=600, margin=dict(l=10, r=10, t=40, b=10))
        return fig
    except Exception as e:
        return None


def plot_forecast(history, validation, forecasts):
    """Строит красивый график прогноза с возможностью отображения нескольких прогнозов."""
    fig = go.Figure()
    
    # Цвета для моделей
    colors = plotly.colors.qualitative.Plotly

    # История
    hist_idx = history.time_index if hasattr(history, 'time_index') else history.index
    hist_val = history.values().flatten() if hasattr(history, 'values') else history.values
    fig.add_trace(go.Scatter(x=hist_idx, y=hist_val, name="История", line=dict(color='gray', width=1)))

    # Валидация
    if validation is not None:
        fig.add_trace(go.Scatter(x=validation.time_index, y=validation.values().flatten(), name="Валидация",
                                 line=dict(color='#FFA500', dash='dot')))

    # Прогнозы
    if forecasts:
        for i, (model_name, forecast_ts) in enumerate(forecasts.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=forecast_ts.time_index, y=forecast_ts.values().flatten(), name=f"Прогноз ({model_name})",
                                     line=dict(color=color, width=2)))

    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
    return fig


def safe_export_df(ts_obj, col_name='Value'):
    """Конвертирует TimeSeries в DataFrame для Excel"""
    try:
        vals = ts_obj.values().flatten()
        idx = ts_obj.time_index
        return pd.DataFrame(data=vals, index=idx, columns=[col_name]).reset_index()
    except:
        return pd.DataFrame(columns=['Error'])


def create_excel_download(history_ts, forecast_ts, metrics_df=None):
    """Генерирует Excel файл в памяти"""
    b = BytesIO()
    with pd.ExcelWriter(b, engine='openpyxl') as w:
        safe_export_df(history_ts, 'History').to_excel(w, sheet_name='Hist', index=False)
        safe_export_df(forecast_ts, 'Forecast').to_excel(w, sheet_name='Fcst', index=False)
        if metrics_df is not None:
            metrics_df.to_excel(w, sheet_name='Leaderboard', index=False)
    b.seek(0)
    return b

def export_fig_to_png(fig):
    """Экспортирует фигуру Plotly в PNG байты."""
    if fig is None:
        return None
    try:
        img_bytes = fig.to_image(format="png", engine="kaleido")
        return BytesIO(img_bytes)
    except Exception as e:
        print(f"Ошибка экспорта в PNG: {e}")
        return None
