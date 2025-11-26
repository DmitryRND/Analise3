import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# Darts imports
from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing,
    LightGBMModel,
    AutoARIMA,
    Theta,
    LinearRegressionModel,
    NaiveDrift
)
from darts.metrics import mae, mse, rmse, mape
from darts.utils.missing_values import fill_missing_values

# Проверка Prophet
PROPHET_AVAILABLE = False
try:
    from darts.models import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(page_title="TS Master v4.0", layout="wide")


# --- ФУНКЦИИ ---

def detect_seasonality_period(df_index):
    """
    Умное определение периода на основе индекса Pandas.
    Возвращает (int Period, str Reasoning)
    """
    freq = pd.infer_freq(df_index)
    if freq:
        freq = freq.upper()
        if 'M' in freq: return 12, "Месячные данные (Detected: Month)"
        if 'Q' in freq: return 4, "Квартальные данные (Detected: Quarter)"
        if 'H' in freq: return 24, "Часовые данные (Detected: Hour)"
        if 'D' in freq: return 7, "Дневные данные (Default: Week)"
        if 'W' in freq: return 52, "Недельные данные (Default: Year)"

    # Fallback если частоту не поняли
    if len(df_index) < 60: return 12, "Мало данных, предполагаем Месяцы"
    return 7, "Частота не определена, предполагаем Недельный цикл"


def check_seasonality(df, value_col, specified_period):
    try:
        # Используем период, который выбрала система или пользователь
        period = int(specified_period)
        if period >= len(df) // 2: period = 2

        decomposition = seasonal_decompose(df[value_col], model='additive', period=period)

        seasonal_var = np.var(decomposition.seasonal)
        resid_var = np.var(decomposition.resid.dropna())

        # Если сезонность объясняет больше вариации, чем шум
        has_seasonality = seasonal_var > (resid_var * 0.1)
        return has_seasonality, decomposition
    except:
        return False, None


def detect_outliers(df, value_col, threshold=3):
    z = np.abs(stats.zscore(df[value_col]))
    outliers = df[z > threshold]
    return outliers, len(outliers) > 0


# --- ИНТЕРФЕЙС ---

st.title("🧠 Time Series Master v4.0 (AI + Stats)")

# 1. ЗАГРУЗКА
st.sidebar.header("1. Данные")
uploaded_file = st.sidebar.file_uploader("Файл (CSV/XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    # Чтение
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    cols = df.columns.tolist()

    # Авто-выбор
    date_guess = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'month' in c.lower()), cols[0])
    target_guess = next((c for c in cols if c != date_guess and pd.api.types.is_numeric_dtype(df[c])),
                        cols[1] if len(cols) > 1 else cols[0])

    c1, c2 = st.sidebar.columns(2)
    date_col = c1.selectbox("Дата", cols, index=cols.index(date_guess))
    target_col = c2.selectbox("Значение", cols, index=cols.index(target_guess))

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        df = df.set_index(date_col)
        df[target_col] = df[target_col].interpolate()  # Заполняем пропуски
    except Exception as e:
        st.error(f"Ошибка даты: {e}")
        st.stop()

    # --- 2. АНАЛИЗ ---
    st.header("2. Анализ ряда")

    # АВТО-ДЕТЕКТ СЕЗОННОСТИ
    auto_period, period_reason = detect_seasonality_period(df.index)

    with st.expander("🔍 Настройки сезонности", expanded=True):
        st.caption(f"Система определила: {period_reason}")
        period_input = st.number_input("Период сезонности (шагов)", min_value=2, value=auto_period)

    has_seasonality, decomposition = check_seasonality(df.reset_index(), target_col, specified_period=period_input)
    outliers_df, has_outliers = detect_outliers(df.reset_index(), target_col)

    # График
    fig_diag = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                             subplot_titles=("Исходный ряд", "Сезонность"))
    fig_diag.add_trace(go.Scatter(x=df.index, y=df[target_col], name='Факт'), row=1, col=1)
    if has_outliers:
        fig_diag.add_trace(
            go.Scatter(x=outliers_df[date_col], y=outliers_df[target_col], mode='markers', name='Выбросы',
                       marker=dict(color='red', symbol='x')), row=1, col=1)
    if decomposition:
        fig_diag.add_trace(
            go.Scatter(x=df.index, y=decomposition.seasonal, name='Сезонность', line=dict(color='green')), row=2, col=1)
    st.plotly_chart(fig_diag, use_container_width=True)

    # --- 3. МОДЕЛИРОВАНИЕ ---
    st.markdown("---")
    st.header("3. Выбор модели")

    c_h, c_m, c_opt = st.columns(3)
    with c_h:
        horizon = st.number_input("Горизонт прогноза", min_value=1, value=int(period_input))
    with c_m:
        # РАСШИРЕННЫЙ СПИСОК МОДЕЛЕЙ
        model_opts = [
            "ExponentialSmoothing (ETS)",
            "AutoARIMA (Stats)",
            "Theta (Stats)",
            "LinearRegression (Trend)",
            "LightGBM (ML)"
        ]
        if PROPHET_AVAILABLE: model_opts.append("Prophet (Facebook)")

        model_name = st.selectbox("Алгоритм", model_opts)
    with c_opt:
        tuning_mode = st.radio("Режим настройки", ["Ручной (Manual)", "AutoML (Optuna)"])

    # ПАРАМЕТРЫ МОДЕЛЕЙ
    params = {}

    with st.expander(f"🛠 Настройки: {model_name}", expanded=True):

        # 1. EXPONENTIAL SMOOTHING
        if "ExponentialSmoothing" in model_name:
            if tuning_mode == "Ручной (Manual)":
                c1, c2 = st.columns(2)
                trend_mode = c1.selectbox("Trend", ["Model Selects", "Additive", "Multiplicative", "None"])
                seas_mode = c2.selectbox("Seasonal", ["Model Selects", "Additive", "Multiplicative", "None"])

                # Преобразование в формат, понятный Darts (None или lowercase string)
                params['trend'] = None if trend_mode == "Model Selects" else (
                    None if trend_mode == "None" else trend_mode.lower())
                params['seasonal'] = None if seas_mode == "Model Selects" else (
                    None if seas_mode == "None" else seas_mode.lower())
            else:
                st.info("Optuna подберет тип тренда и сезонности.")

        # 2. AUTO ARIMA
        elif "AutoARIMA" in model_name:
            st.info("AutoARIMA автоматически подбирает параметры (p,d,q). Это может занять время.")
            # AutoARIMA почти не требует ручных настроек для базового использования

        # 3. THETA
        elif "Theta" in model_name:
            if tuning_mode == "Ручной (Manual)":
                theta_param = st.number_input("Theta Parameter (0=Linear, 2=Standard)", value=2.0)
                params['theta'] = theta_param
            else:
                st.info("Optuna подберет параметр Theta.")

        # 4. LINEAR REGRESSION
        elif "LinearRegression" in model_name:
            st.info("Строит линейный тренд + лаги. Хорошо для данных с явным ростом/падением.")
            lags_lr = st.slider("Lags (учитывать прошлые N точек)", 1, 60, 12)
            params['lags'] = lags_lr

        # 5. LIGHTGBM
        elif "LightGBM" in model_name:
            st.warning(
                "LightGBM плохо экстраполирует тренды. Используйте его для стационарных данных или уберите тренд.")
            lags_input = st.slider("Lags", 1, 60, 12)
            params['lags'] = lags_input
            if tuning_mode == "Ручной (Manual)":
                lr_input = st.number_input("Learning Rate", 0.001, 0.5, 0.05, step=0.01)
                params['learning_rate'] = lr_input

        # 6. PROPHET
        elif "Prophet" in model_name:
            if tuning_mode == "Ручной (Manual)":
                col_p1, col_p2 = st.columns(2)
                seasonality_mode = col_p1.selectbox("Seasonality Mode", ["additive", "multiplicative"])
                # Добавлены новые настройки
                changepoint_scale = col_p2.slider("Гибкость тренда (Changepoint Scale)", 0.001, 0.5, 0.05)

                params['seasonality_mode'] = seasonality_mode
                params['changepoint_prior_scale'] = changepoint_scale
            else:
                st.info("Optuna подберет режим сезонности и гибкость тренда.")

    # --- ЗАПУСК ---
    if st.button("🚀 Выполнить прогноз"):

        # Подготовка данных
        ts = TimeSeries.from_dataframe(df.reset_index(), time_col=date_col, value_cols=target_col)
        ts = fill_missing_values(ts)

        # Сплит
        val_len = horizon if horizon < len(ts) * 0.3 else int(len(ts) * 0.2)
        train, val = ts.split_before(len(ts) - val_len)
        metric_func = mae

        model_obj = None

        with st.spinner('Обучение модели... (AutoARIMA может думать долго)'):

            # === AUTOML (OPTUNA) ===
            if tuning_mode == "AutoML (Optuna)":
                def objective(trial):
                    m = None
                    if "ExponentialSmoothing" in model_name:
                        t = trial.suggest_categorical("trend", [None, "additive", "multiplicative"])
                        s = trial.suggest_categorical("seasonal", [None, "additive", "multiplicative"])
                        m = ExponentialSmoothing(trend=t, seasonal=s, seasonal_periods=period_input)
                    elif "Theta" in model_name:
                        th = trial.suggest_float("theta", 0, 5)
                        m = Theta(theta=th,
                                  season_mode=trial.suggest_categorical("mode", ["additive", "multiplicative"]))
                    elif "LightGBM" in model_name:
                        l = trial.suggest_int("lags", 4, 30)
                        lr = trial.suggest_float("learning_rate", 0.01, 0.3)
                        m = LightGBMModel(lags=l, learning_rate=lr, output_chunk_length=1, verbose=-1)
                    elif "Prophet" in model_name:
                        sm = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
                        cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5)
                        m = Prophet(seasonality_mode=sm, changepoint_prior_scale=cps)
                    elif "LinearRegression" in model_name:
                        l = trial.suggest_int("lags", 1, 30)
                        m = LinearRegressionModel(lags=l)
                    elif "AutoARIMA" in model_name:
                        # AutoARIMA не тюним через Optuna, она сама тюнится
                        m = AutoARIMA()

                    m.fit(train)
                    p = m.predict(len(val))
                    return mae(val, p)


                # Для AutoARIMA пропускаем Optuna
                if "AutoARIMA" in model_name:
                    best_p = {}
                else:
                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=10)  # 10 попыток для скорости
                    best_p = study.best_params
                    st.success(f"Optuna нашла: {best_p}")

                # Инициализация лучшей модели
                if "ExponentialSmoothing" in model_name:
                    model_obj = ExponentialSmoothing(trend=best_p.get('trend'), seasonal=best_p.get('seasonal'),
                                                     seasonal_periods=period_input)
                elif "Theta" in model_name:
                    model_obj = Theta(theta=best_p.get('theta'), season_mode=best_p.get('mode', 'multiplicative'))
                elif "LightGBM" in model_name:
                    model_obj = LightGBMModel(lags=best_p['lags'], learning_rate=best_p['learning_rate'],
                                              output_chunk_length=1)
                elif "Prophet" in model_name:
                    model_obj = Prophet(seasonality_mode=best_p['seasonality_mode'],
                                        changepoint_prior_scale=best_p['changepoint_prior_scale'])
                elif "LinearRegression" in model_name:
                    model_obj = LinearRegressionModel(lags=best_p['lags'])
                elif "AutoARIMA" in model_name:
                    model_obj = AutoARIMA()

            # === MANUAL MODE ===
            else:
                if "ExponentialSmoothing" in model_name:
                    model_obj = ExponentialSmoothing(trend=params['trend'], seasonal=params['seasonal'],
                                                     seasonal_periods=period_input)
                elif "AutoARIMA" in model_name:
                    model_obj = AutoARIMA()
                elif "Theta" in model_name:
                    model_obj = Theta(theta=params['theta'])
                elif "LinearRegression" in model_name:
                    model_obj = LinearRegressionModel(lags=params['lags'])
                elif "LightGBM" in model_name:
                    model_obj = LightGBMModel(lags=params['lags'], learning_rate=params['learning_rate'],
                                              output_chunk_length=1)
                elif "Prophet" in model_name:
                    model_obj = Prophet(seasonality_mode=params['seasonality_mode'],
                                        changepoint_prior_scale=params['changepoint_prior_scale'])

            # ОБУЧЕНИЕ
            model_obj.fit(train)
            pred_val = model_obj.predict(len(val))
            score = mae(val, pred_val)

            # ПРОГНОЗ В БУДУЩЕЕ
            model_obj.fit(ts)  # Refit on full data
            pred_future = model_obj.predict(horizon)

            # ГРАФИК
            fig_res = go.Figure()
            fig_res.add_trace(
                go.Scatter(x=ts.time_index, y=ts.values().flatten(), name="История", line=dict(color='gray')))
            fig_res.add_trace(go.Scatter(x=val.time_index, y=pred_val.values().flatten(), name="Валидация",
                                         line=dict(color='orange', dash='dot')))
            fig_res.add_trace(go.Scatter(x=pred_future.time_index, y=pred_future.values().flatten(), name="ПРОГНОЗ",
                                         line=dict(color='green', width=3)))

            st.plotly_chart(fig_res, use_container_width=True)
            st.metric("Качество (MAE)", f"{score:.4f}")

            # ЭКСПОРТ
            try:
                df_hist = ts.pd_dataframe().reset_index()
                df_pred = pred_future.pd_dataframe().reset_index()
                df_pred.columns = [date_col, 'Forecast_Value']

                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_hist.to_excel(writer, sheet_name='History', index=False)
                    df_pred.to_excel(writer, sheet_name='Forecast', index=False)

                buffer.seek(0)
                st.download_button("📥 Скачать Excel", data=buffer, file_name="forecast_v4.xlsx")
            except Exception as e:
                st.error(f"Excel Error: {e}")

else:
    st.info("Загрузите файл (CSV/XLSX)")