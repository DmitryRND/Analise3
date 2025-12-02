import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from io import BytesIO
from darts import TimeSeries
import plotly.colors
import plotly.io as pio
pio.templates.default = "plotly_dark"

def _get_ts_values_and_index(ts: TimeSeries):
    """Вспомогательная функция для получения значений и индекса из TimeSeries."""
    try:
        # Пробуем использовать values() метод
        if hasattr(ts, 'values') and callable(getattr(ts, 'values')):
            values = ts.values().flatten()
        elif hasattr(ts, 'data_array'):
            values = ts.data_array().values.flatten()
        elif hasattr(ts, 'pd_dataframe'):
            df = ts.pd_dataframe()
            values = df.iloc[:, 0].values
        elif hasattr(ts, 'to_dataframe'):
            df = ts.to_dataframe()
            values = df.iloc[:, 0].values
        else:
            # Последний вариант - через values_at_times
            values = np.array([ts.values_at_times(ts.time_index[i])[0] for i in range(len(ts))])
        
        return values, ts.time_index
    except Exception as e:
        # Финальный вариант - создаем Series из всех возможных методов
        try:
            df = pd.DataFrame(ts.values().T, index=ts.time_index)
            values = df.iloc[:, 0].values
            return values, ts.time_index
        except:
            raise ValueError(f"Не удалось извлечь данные из TimeSeries: {e}")


def _get_ts_dataframe(ts: TimeSeries):
    """Вспомогательная функция для конвертации TimeSeries в pandas DataFrame."""
    try:
        if hasattr(ts, "pd_dataframe"):
            return ts.pd_dataframe()
        if hasattr(ts, "to_dataframe"):
            return ts.to_dataframe()
    except Exception:
        pass
    values, index = _get_ts_values_and_index(ts)
    return pd.DataFrame({'value': values}, index=index)

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
    Анализирует временной ряд и рекомендует метрику (MAE, MAPE, RMSE или MSE)
    с пояснением.
    """
    values, index = _get_ts_values_and_index(ts)
    series = pd.Series(values, index=index)

    # 1. Проверка на нули или отрицательные значения
    if (series <= 0).any():
        return {
            "metric": "MAE",
            "reason": "В данных есть нулевые или отрицательные значения. "
                      "MAE измеряет ошибку в тех же единицах и устойчив к таким значениям."
        }

    # 2. Проверка на выбросы
    if len(series) > 10:
        z_scores = np.abs(stats.zscore(series))
        if np.any(z_scores > 3):
            return {
                "metric": "MAE",
                "reason": "Выявлены значительные выбросы. MAE менее чувствителен к экстремальным значениям."
            }

    # 3. Оценка вариабельности: коэффициент вариации
    mean_val = series.mean()
    std_val = series.std()
    coeff_var = std_val / mean_val if mean_val != 0 else np.inf

    # 4. Выбор метрики на основе вариабельности
    if coeff_var < 0.1:
        # Данные стабильные, можно использовать MAPE
        return {
            "metric": "MAPE",
            "reason": "Данные стабильны, без выбросов и отрицательных значений. "
                      "MAPE показывает ошибку в процентах, что удобно для интерпретации."
        }
    elif coeff_var < 0.5:
        # Средняя вариабельность — рекомендуем RMSE
        return {
            "metric": "RMSE",
            "reason": "В данных нет нулей и выбросов, но наблюдается заметная вариабельность. "
                      "RMSE удобно интерпретировать, и она увеличивает влияние крупных ошибок, что полезно в этом случае."
        }
    else:
        # Большая вариабельность — используем MSE
        return {
            "metric": "MSE",
            "reason": "В данных высокая вариабельность. "
                      "MSE строго штрафует большие ошибки благодаря возведению отклонения в квадрат, "
                      "что помогает уделить больше внимания крупным промахам [oai_citation:8‡jedox.com](https://www.jedox.com/en/blog/error-metrics-how-to-evaluate-forecasts/#:~:text=problem%2C%20Mean%20Square%20Error%20can,to%20deal%20with%20that%20situation)."
        }

    return {
        "metric": "MAPE",
        "reason": "Ваши данные выглядят стабильными, без нулей и сильных выбросов. "
                  "Рекомендуется использовать MAPE (средняя абсолютная процентная ошибка), "
                  "так как она показывает ошибку в процентах, что облегчает интерпретацию качества модели."
    }


def plot_decomposition(ts: TimeSeries, value_col: str = None, period: int = 12):
    """Строит график декомпозиции (Тренд, Сезонность, Шум) с выделением выпадов/аномалий"""
    if len(ts) < period * 2:
        raise ValueError(f"Для анализа сезонности необходимо как минимум {period * 2} точек данных (2 периода), а у вас {len(ts)}.")
    try:
        # Получаем значения из TimeSeries
        values, index = _get_ts_values_and_index(ts)
        series = pd.Series(values, index=index)
        
        # Обрабатываем пропущенные значения перед декомпозицией
        # Удаляем только полностью отсутствующие значения
        series_clean = series.dropna()
        
        if len(series_clean) < period * 2:
            raise ValueError(f"После очистки от пропусков осталось только {len(series_clean)} точек, что меньше необходимых {period * 2}.")
        
        # Если есть пропуски, используем интерполяцию для непрерывного ряда
        if len(series_clean) < len(series):
            # Переиндексируем для непрерывного временного ряда
            if isinstance(series_clean.index, pd.DatetimeIndex):
                full_index = pd.date_range(start=series_clean.index.min(), end=series_clean.index.max(), freq=pd.infer_freq(series_clean.index) or 'D')
                series_clean = series_clean.reindex(full_index)
                series_clean = series_clean.interpolate(method='time')
        
        # Декомпозиция (теперь без пропусков)
        decomposition = seasonal_decompose(series_clean, model='additive', period=period, extrapolate_trend='freq')

        # Обнаружение выпадов (аномалий) в остатках
        resid = decomposition.resid.dropna()
        if len(resid) > 0:
            z_scores = np.abs(stats.zscore(resid))
            outliers_mask = z_scores > 3
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=("Исходный ряд", "Тренд", "Сезонность", "Остатки (выпады выделены красным)"),
                            vertical_spacing=0.08)

        # Исходный ряд
        fig.add_trace(go.Scatter(x=series.index, y=series, name='Исходный ряд', line=dict(color='#1f77b4', width=1)), row=1, col=1)
        
        # Тренд
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Тренд', line=dict(color='#FF9900', width=2)), row=2, col=1)
        
        # Сезонность
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Сезонность', line=dict(color='#00CC66', width=2)), row=3, col=1)
        
        # Остатки с выделением выпадов
        fig.add_trace(go.Scatter(x=resid.index, y=resid, name='Остатки', mode='lines+markers', 
                                marker=dict(color='gray', size=4), line=dict(color='gray', width=1)), row=4, col=1)
        
        # Выделение выпадов красным
        if len(resid) > 0 and np.any(outliers_mask):
            outlier_resid = resid[outliers_mask]
            outlier_indices = outlier_resid.index
            fig.add_trace(go.Scatter(x=outlier_indices, y=outlier_resid, name='Выпады', 
                                    mode='markers', marker=dict(color='red', size=8, symbol='x')), row=4, col=1)

        fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        return fig
    except Exception as e:
        raise ValueError(f"Не удалось построить график декомпозиции: {e}")


def plot_backtest(history: TimeSeries, validation: TimeSeries, forecasts: dict):
    """Строит график сравнения прогнозов моделей на тестовых данных."""
    fig = go.Figure()
    palette = [
        "#4dabf7", "#f59f00", "#20c997", "#e64980", "#ffa94d",
        "#94d82d", "#748ffc", "#ff922b", "#63e6be", "#ced4da",
    ]
    
    # Получаем данные из TimeSeries
    hist_values, hist_index = _get_ts_values_and_index(history)
    val_values, val_index = _get_ts_values_and_index(validation)
    
    fig.add_trace(go.Scatter(x=hist_index, y=hist_values, name="История", line=dict(color='#E6E8EF', width=2)))
    fig.add_trace(go.Scatter(x=val_index, y=val_values, name="Тестовые данные", line=dict(color='#FF6B6B', dash='dash')))

    for i, (model_name, forecast_ts) in enumerate(forecasts.items()):
        if forecast_ts is not None:
            color = palette[i % len(palette)]
            forecast_values, forecast_index = _get_ts_values_and_index(forecast_ts)
            fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, name=model_name, line=dict(color=color, width=2)))

    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Значение")
    return fig


def plot_forecast(history: TimeSeries, validation: TimeSeries, forecasts: dict):
    """Алиас для plot_backtest - строит график сравнения прогнозов моделей на тестовых данных."""
    return plot_backtest(history, validation, forecasts)


def plot_final_forecast(history: TimeSeries, forecast: TimeSeries):
    """Строisceт график финального прогноза от лучшей модели."""
    fig = go.Figure()
    
    # Получаем данные из TimeSeries
    hist_values, hist_index = _get_ts_values_and_index(history)
    forecast_values, forecast_index = _get_ts_values_and_index(forecast)
    
    fig.add_trace(go.Scatter(x=hist_index, y=hist_values, name="Вся история", line=dict(color='#E6E8EF', width=2)))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_values, name="Финальный прогноз", line=dict(color='#4dabf7', width=3)))
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Значение")
    return fig


def create_excel_download(ts_or_df):
    """Генерирует Excel файл из DataFrame или TimeSeries."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(ts_or_df, pd.DataFrame):
            ts_or_df.to_excel(writer, sheet_name='Data', index=True)
        else:
            # Если передан TimeSeries - конвертируем в DataFrame
            df = _get_ts_dataframe(ts_or_df)
            df.to_excel(writer, sheet_name='Data', index=True)
    output.seek(0)
    return output.getvalue()
