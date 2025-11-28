import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from io import BytesIO
from darts import TimeSeries
import plotly.colors

def load_timeseries_from_file(file, date_col=None, value_col=None, freq='D'):
    """
    Загружает временной ряд из файла (CSV или Excel), автоматически определяет дату
    или использует указанные столбцы.
    """
    try:
        try:
            df = pd.read_csv(file)
        except Exception:
            file.seek(0)
            df = pd.read_excel(file)
    except Exception as e:
        return None, f"Не удалось прочитать файл. Ошибка: {e}"

    if date_col is None:
        # Попытка автоопределения колонки с датой
        for col in df.columns:
            if df[col].dtype in ['datetime64[ns]', 'object']:
                try:
                    # Пробуем преобразовать, если успешно на >80% строк - используем
                    pd.to_datetime(df[col], errors='raise')
                    date_col = col
                    break
                except (ValueError, TypeError):
                    continue
    
    if date_col is None:
        return df, "Не удалось найти столбец с датой. Пожалуйста, укажите его вручную."

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        return df, f"Ошибка преобразования столбца '{date_col}' в дату. Ошибка: {e}"

    df = df.set_index(date_col).sort_index()

    # Если value_col не указан, берем первый числовой столбец
    if value_col is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return df, "В файле нет числовых столбцов для прогнозирования."
        value_col = numeric_cols[0]

    # Убираем дубликаты в индексе
    df = df[~df.index.duplicated(keep='first')]
    
    # Создаем TimeSeries, заполняя пропуски, если они есть
    ts = TimeSeries.from_dataframe(df, value_cols=[value_col], freq=freq, fill_missing_dates=True, ffill=True)
    
    return ts, None # Возвращаем TimeSeries и отсутствие ошибки


def recommend_metric(ts: TimeSeries):
    """
    Анализирует временной ряд и рекомендует метрику (MAE или MAPE)
    с объяснением.
    """
    series = ts.pd_series()
    # Проверка на наличие нулей или отрицательных значений
    if (series <= 0).any():
        return {
            "metric": "MAE",
            "reason": "В ваших данных присутствуют нулевые или отрицательные значения. "
                      "MAPE (процентная ошибка) не может быть корректно рассчитана в таких случаях. "
                      "Рекомендуется использовать MAE (средняя абсолютная ошибка), которая измеряет ошибку в тех же единицах, что и ваши данные."
        }
    
    # Проверка на выбросы
    if len(series) > 10:
        z_scores = np.abs(stats.zscore(series))
        if np.any(z_scores > 3):
            return {
                "metric": "MAE",
                "reason": "В ваших данных обнаружены значительные выбросы. "
                          "MAPE может быть искажена из-за них. "
                          "Рекомендуется использовать MAE (средняя абсолютная ошибка), так как она менее чувствительна к экстремальным значениям."
            }

    return {
        "metric": "MAPE",
        "reason": "Ваши данные выглядят стабильными, без нулей и сильных выбросов. "
                  "Рекомендуется использовать MAPE (средняя абсолютная процентная ошибка), "
                  "так как она показывает ошибку в процентах, что облегчает интерпретацию качества модели."
    }


def plot_decomposition(ts: TimeSeries, period: int):
    """Строит график декомпозиции (Тренд, Сезонность, Шум)"""
    if len(ts) < period * 2:
        return None, "Для анализа сезонности необходимо как минимум два полных периода данных."
    try:
        df = ts.pd_dataframe()
        decomposition = seasonal_decompose(df[df.columns[0]], model='additive', period=period)

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Тренд", "Сезонность", "Остатки (Шум)"),
                            vertical_spacing=0.1)

        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Тренд', line=dict(color='#FF9900')), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Сезонность', line=dict(color='#00CC66')), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Остатки', mode='markers', marker=dict(color='gray', size=3)), row=3, col=1)

        fig.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        return fig, None
    except Exception as e:
        return None, f"Не удалось построить график декомпозиции. Ошибка: {e}"


def plot_backtest(history: TimeSeries, validation: TimeSeries, forecasts: dict):
    """Строит график сравнения прогнозов моделей на тестовых данных."""
    fig = go.Figure()
    colors = plotly.colors.qualitative.Plotly

    fig.add_trace(go.Scatter(x=history.pd_dataframe().index, y=history.pd_series(), name="История", line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=validation.pd_dataframe().index, y=validation.pd_series(), name="Тестовые данные", line=dict(color='#FF4136', dash='dash')))

    for i, (model_name, forecast_ts) in enumerate(forecasts.items()):
        if forecast_ts is not None:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=forecast_ts.pd_dataframe().index, y=forecast_ts.pd_series(), name=model_name, line=dict(color=color, width=2)))

    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Значение")
    return fig


def plot_final_forecast(history: TimeSeries, forecast: TimeSeries):
    """Строisceт график финального прогноза от лучшей модели."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.pd_dataframe().index, y=history.pd_series(), name="Вся история", line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=forecast.pd_dataframe().index, y=forecast.pd_series(), name="Финальный прогноз", line=dict(color='#0074D9', width=3)))
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Значение")
    return fig


def create_excel_download(history_ts, validation_ts, forecasts, final_forecast, metrics_df):
    """Генерирует Excel файл со всеми данными."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if metrics_df is not None:
            metrics_df.to_excel(writer, sheet_name='Leaderboard', index=False)

        full_actual_df = pd.concat([history_ts.pd_dataframe(), validation_ts.pd_dataframe()])
        full_actual_df.columns = ['Actual']
        
        forecasts_df = pd.DataFrame(index=full_actual_df.index)
        if forecasts:
            for model_name, ts in forecasts.items():
                if ts:
                    forecasts_df[f'Backtest_{model_name}'] = ts.pd_series()

        if final_forecast:
            forecasts_df[f'Final_Forecast'] = final_forecast.pd_series()
            
        combined_df = pd.concat([full_actual_df, forecasts_df], axis=1)
        combined_df.reset_index().to_excel(writer, sheet_name='All_Data', index=False)
        
    output.seek(0)
    return output

def create_png_download(fig):
    """Конвертирует фигуру Plotly в PNG для скачивания."""
    output = BytesIO()
    fig.write_image(output, format="png", scale=2, width=1000, height=600)
    output.seek(0)
    return output
