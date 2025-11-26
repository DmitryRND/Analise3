import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

# --- DARTS IMPORTS ---
from darts import TimeSeries
from darts.models import (
    ExponentialSmoothing,
    LightGBMModel,
    Theta,
    LinearRegressionModel,
    Prophet,
    ARIMA
)
from darts.metrics import mae, mse, rmse, mape
from darts.utils.missing_values import fill_missing_values
from darts.utils.utils import ModelMode, SeasonalityMode

# AutoARIMA Check
AUTOARIMA_AVAILABLE = False
try:
    from darts.models import AutoARIMA

    AUTOARIMA_AVAILABLE = True
except ImportError:
    pass

# --- CONFIG & CSS ---
st.set_page_config(page_title="TS Master v12.0", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #00CC66;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #00994d;
        border: 1px solid white;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00CC66;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- DICTIONARIES (HELPERS) ---

MODEL_DESCRIPTIONS = {
    "ARIMA (Optuna)": "🧠 **Smart ARIMA**: Классическая статистика, но параметры (p,d,q) подбирает ИИ. Отлично для трендов и циклов.",
    "AutoARIMA (Classic)": "⚡ **AutoARIMA**: Быстрый стандартный алгоритм подбора параметров по критерию AIC.",
    "ExponentialSmoothing": "📉 **ETS**: Экспоненциальное сглаживание. Идеально для данных с четкой сезонностью и трендом.",
    "Prophet": "📅 **Prophet**: Модель от Facebook. Устойчива к выбросам, праздникам и сменам тренда.",
    "LightGBM": "🌳 **LightGBM**: Мощный ML-алгоритм (градиентный бустинг). Лучше всего работает, когда данных много.",
    "Theta": "🔄 **Theta**: Простой, но удивительно точный метод. Часто побеждает сложные модели.",
    "LinearRegression": "📏 **LinReg**: Простая линейная регрессия. Хороша как базовый уровень (Baseline).",
    "ARIMA (Manual)": "🛠 **Manual ARIMA**: Для экспертов. Вы сами задаете порядки p, d, q."
}

METRIC_DESCRIPTIONS = {
    "MAE": "Средняя абсолютная ошибка. Легко интерпретировать (в тех же единицах, что и данные). Устойчива к выбросам.",
    "RMSE": "Корень из среднеквадратичной ошибки. Сильно штрафует за большие промахи. Хорошо, если критичны пиковые ошибки.",
    "MAPE": "Средняя процентная ошибка. Понятна бизнесу (ошибка в %). Не работает, если есть нули в данных.",
    "MSE": "Среднеквадратичная ошибка. Используется математиками для оптимизации."
}


# --- HELPERS ---

def get_safe_lags(data_len):
    max_lags = int(data_len / 2) - 1
    return min(12, max_lags) if max_lags > 1 else 1


def map_model_mode(val_str):
    if val_str == "additive": return ModelMode.ADDITIVE
    if val_str == "multiplicative": return ModelMode.MULTIPLICATIVE
    return ModelMode.NONE


def map_seasonality_mode(val_str):
    if val_str == "additive": return SeasonalityMode.ADDITIVE
    if val_str == "multiplicative": return SeasonalityMode.MULTIPLICATIVE
    return SeasonalityMode.ADDITIVE


def check_seasonality(df, value_col, period):
    try:
        if period < 2 or period > len(df) // 2: return False, None
        decomposition = seasonal_decompose(df[value_col], model='additive', period=int(period))
        return True, decomposition
    except:
        return False, None


def detect_outliers_zscore(df, value_col):
    z = np.abs(stats.zscore(df[value_col]))
    return np.any(z > 3)  # True если есть выбросы > 3 сигм


def safe_export_df(ts_obj, col_name='Value'):
    try:
        vals = ts_obj.values().flatten()
        idx = ts_obj.time_index
        return pd.DataFrame(data=vals, index=idx, columns=[col_name]).reset_index()
    except:
        return pd.DataFrame(columns=['Error'])


# --- OPTUNA CORE ---

def run_optimization(model_name, train, val, metric_func, period, n_trials=10):
    def objective(trial):
        try:
            m_tmp = None
            if model_name == "ARIMA (Optuna)":
                p = trial.suggest_int("p", 0, 4)
                d = trial.suggest_int("d", 0, 2)
                q = trial.suggest_int("q", 0, 4)
                is_s = trial.suggest_categorical("seasonal", [True, False])
                so = (trial.suggest_int("P", 0, 2), trial.suggest_int("D", 0, 1), trial.suggest_int("Q", 0, 2),
                      period) if is_s else (0, 0, 0, 0)
                m_tmp = ARIMA(p=p, d=d, q=q, seasonal_order=so)
            elif model_name == "ExponentialSmoothing":
                tr = trial.suggest_categorical("trend", ["additive", "multiplicative", "none"])
                se = trial.suggest_categorical("seasonal", ["additive", "multiplicative", "none"])
                m_tmp = ExponentialSmoothing(trend=map_model_mode(tr), seasonal=map_model_mode(se),
                                             seasonal_periods=period)
            elif model_name == "Theta":
                th = trial.suggest_float("theta", 0.5, 4.0)
                m_tmp = Theta(theta=th)
            elif model_name == "LightGBM":
                l = trial.suggest_int("lags", 6, 30)
                lr = trial.suggest_float("lr", 0.01, 0.3)
                m_tmp = LightGBMModel(lags=l, learning_rate=lr, output_chunk_length=1, verbose=-1)
            elif model_name == "Prophet":
                sm = trial.suggest_categorical("sm", ["additive", "multiplicative"])
                cp = trial.suggest_float("cp", 0.01, 0.5)
                m_tmp = Prophet(seasonality_mode=sm, changepoint_prior_scale=cp)
            elif model_name == "LinearRegression":
                l = trial.suggest_int("lags", 4, 40)
                m_tmp = LinearRegressionModel(lags=l)

            m_tmp.fit(train)
            p = m_tmp.predict(len(val))
            return metric_func(val, p)
        except:
            return float('inf')

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    bp = study.best_params

    # Rebuild
    final_m = None
    if model_name == "ARIMA (Optuna)":
        is_s = bp['seasonal']
        so = (bp['P'], bp['D'], bp['Q'], period) if is_s else (0, 0, 0, 0)
        final_m = ARIMA(p=bp['p'], d=bp['d'], q=bp['q'], seasonal_order=so)
    elif model_name == "ExponentialSmoothing":
        final_m = ExponentialSmoothing(trend=map_model_mode(bp['trend']), seasonal=map_model_mode(bp['seasonal']),
                                       seasonal_periods=period)
    elif model_name == "Theta":
        final_m = Theta(theta=bp['theta'])
    elif model_name == "LightGBM":
        final_m = LightGBMModel(lags=bp['lags'], learning_rate=bp['lr'], output_chunk_length=1)
    elif model_name == "Prophet":
        final_m = Prophet(seasonality_mode=bp['sm'], changepoint_prior_scale=bp['cp'])
    elif model_name == "LinearRegression":
        final_m = LinearRegressionModel(lags=bp['lags'])

    return final_m, bp


# --- UI START ---

st.title("🛡️ TS Master v12.0 (AI Assistant)")

# 1. LOAD
with st.sidebar:
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Файл (CSV/XLSX)", type=['csv', 'xlsx'])
    df = None
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        cols = df.columns.tolist()
        d_col = st.selectbox("Дата", cols, index=0)
        t_col = st.selectbox("Значение", cols, index=1)

if df is not None:
    # Preprocess
    try:
        df[d_col] = pd.to_datetime(df[d_col])
        df = df.sort_values(by=d_col).set_index(d_col)
        df[t_col] = df[t_col].interpolate()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Stats
    freq_detected = pd.infer_freq(df.index)
    rec_period = 12
    if freq_detected and 'D' in freq_detected: rec_period = 7
    has_outliers = detect_outliers_zscore(df, t_col)

    # 2. RECOMMENDATIONS & SETTINGS
    st.header("2. Настройки и Рекомендации")

    # Recommendation Logic
    rec_metric = "MAE" if has_outliers else "RMSE"
    rec_reason = "Обнаружены выбросы! MAE устойчивее." if has_outliers else "Данные чистые. RMSE/MSE подойдут."

    # Info Block
    st.markdown(f"""
    <div class="info-box">
        <b>💡 Анализ данных:</b><br>
        • Частота: {freq_detected if freq_detected else "Не определена"}<br>
        • Рекомендуемый период: <b>{rec_period}</b><br>
        • Рекомендуемая метрика: <b>{rec_metric}</b> ({rec_reason})
    </div>
    """, unsafe_allow_html=True)

    # Common Settings
    c1, c2, c3 = st.columns(3)
    horizon = c1.number_input("📅 Горизонт прогноза", 1, 1000, 12)
    period_in = c2.number_input("🔄 Период сезонности", 2, 365, rec_period)
    metric_in = c3.selectbox("🎯 Метрика", ["MAE", "RMSE", "MSE", "MAPE"],
                             index=["MAE", "RMSE", "MSE", "MAPE"].index(rec_metric),
                             help=METRIC_DESCRIPTIONS[rec_metric])

    # --- 3. PATH SELECTION ---
    st.markdown("---")
    st.subheader("3. Выберите режим работы")

    mode = st.radio("Mode", ["🥊 Битва моделей (Автоматическое сравнение)", "🛠 Подобрать модель вручную"],
                    label_visibility="collapsed", horizontal=True)

    # ---------------- BATTLE MODE ----------------
    if "Битва" in mode:
        st.caption("Система сама обучит 6 разных моделей и выберет лучшую.")

        bc1, bc2 = st.columns([1, 3])
        n_trials_battle = bc1.select_slider("Точность (попыток Optuna)", options=[5, 10, 20, 50], value=10)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🥊 НАЧАТЬ БИТВУ"):
            ts = TimeSeries.from_dataframe(df.reset_index(), time_col=d_col, value_cols=t_col)
            ts = fill_missing_values(ts)

            val_len = horizon if horizon < len(ts) * 0.3 else int(len(ts) * 0.2)
            train, val = ts.split_before(len(ts) - val_len)
            safe_lags = get_safe_lags(len(train))
            metric_func = {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape}[metric_in]

            status = st.status("Идет соревнование...", expanded=True)

            # Fighters
            fighters = [
                ("ExponentialSmoothing", ExponentialSmoothing(seasonal_periods=period_in)),
                ("Theta", Theta()),
                ("LinearRegression", LinearRegressionModel(lags=safe_lags)),
                ("LightGBM", LightGBMModel(lags=safe_lags, output_chunk_length=1, verbose=-1)),
                ("Prophet", Prophet()),
                ("ARIMA (Optuna)", None)
            ]
            if AUTOARIMA_AVAILABLE: fighters.append(("AutoARIMA", AutoARIMA(seasonal=True)))

            results = []

            for m_name, def_model in fighters:
                status.write(f"Обучение: **{m_name}**...")
                try:
                    final_m = def_model
                    params_txt = "Default/Auto"

                    # Optimization for deep battle (AutoARIMA skipped)
                    if m_name != "AutoARIMA":
                        final_m, bp = run_optimization(m_name, train, val, metric_func, period_in,
                                                       n_trials=n_trials_battle)
                        params_txt = str(bp)

                    final_m.fit(train)
                    pred = final_m.predict(len(val))
                    score = metric_func(val, pred)

                    results.append({
                        "Model": m_name, "Score": score,
                        "Obj": final_m, "Pred": pred, "Params": params_txt,
                        "MAE": mae(val, pred), "MAPE": mape(val, pred)
                    })
                except Exception as e:
                    pass  # st.error(e)

            status.update(label="Битва завершена!", state="complete", expanded=False)

            if results:
                res_df = pd.DataFrame(results).sort_values("Score")
                best = res_df.iloc[0]

                st.success(f"🏆 Победитель: **{best['Model']}** (Ошибка: {best['Score']:.4f})")
                st.dataframe(res_df[["Model", "Score", "MAE", "MAPE", "Params"]].style.highlight_min(subset=["Score"],
                                                                                                     color='#d1e7dd'))

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=val.time_index, y=val.values().flatten(), name="ФАКТ",
                                         line=dict(color='black', width=3)))
                for r in results:
                    is_best = r['Model'] == best['Model']
                    width = 4 if is_best else 1
                    op = 1.0 if is_best else 0.3
                    fig.add_trace(
                        go.Scatter(x=r['Pred'].time_index, y=r['Pred'].values().flatten(), name=r['Model'], opacity=op,
                                   line=dict(width=width)))
                st.plotly_chart(fig, use_container_width=True)

                # Forecast
                with st.spinner("Финальный прогноз..."):
                    best['Obj'].fit(ts)
                    fcst = best['Obj'].predict(horizon)

                fig_f = go.Figure()
                fig_f.add_trace(
                    go.Scatter(x=ts.time_index, y=ts.values().flatten(), name="История", line=dict(color='gray')))
                fig_f.add_trace(go.Scatter(x=fcst.time_index, y=fcst.values().flatten(), name="ПРОГНОЗ",
                                           line=dict(color='#00CC66', width=3)))
                st.plotly_chart(fig_f, use_container_width=True)

                # Export
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w:
                    safe_export_df(ts, 'Hist').to_excel(w, sheet_name='Hist', index=False)
                    safe_export_df(fcst, 'Fcst').to_excel(w, sheet_name='Fcst', index=False)
                    res_df[["Model", "Score", "Params"]].to_excel(w, sheet_name='Leaderboard', index=False)
                b.seek(0)
                st.download_button("📥 Результаты (XLSX)", b, "battle.xlsx")


    # ---------------- MANUAL MODE ----------------
    else:
        mc1, mc2 = st.columns([2, 1])

        # Model List
        m_opts = ["ARIMA (Optuna)", "ExponentialSmoothing", "LightGBM", "Prophet", "Theta", "LinearRegression",
                  "ARIMA (Manual)"]
        if AUTOARIMA_AVAILABLE: m_opts.insert(1, "AutoARIMA (Classic)")

        sel_model = mc1.selectbox("Выберите модель", m_opts)

        # Model Description Info
        st.info(MODEL_DESCRIPTIONS.get(sel_model, "Стандартная модель."))

        # Configuration
        use_optuna = False
        n_trials_man = 10
        manual_params = {}

        # Special cases where Optuna is NOT allowed or ALWAYS on
        if "AutoARIMA" in sel_model:
            st.caption("Работает автоматически (AIC).")
        elif "ARIMA (Optuna)" in sel_model:
            use_optuna = True
            n_trials_man = mc2.select_slider("Попыток", [10, 20, 50], value=10)
        elif "Manual" in sel_model:
            st.caption("Только ручные настройки.")
        else:
            # Standard models: User choice
            use_optuna = mc2.checkbox("Использовать Optuna?", value=False)
            if use_optuna:
                n_trials_man = mc2.select_slider("Попыток", [10, 20, 50, 100], value=20)
            else:
                # Manual Inputs Render
                with st.expander("🎛 Настройки параметров", expanded=True):
                    if "Exponential" in sel_model:
                        c1, c2 = st.columns(2)
                        t = c1.selectbox("Trend", ["additive", "multiplicative", "none"])
                        s = c2.selectbox("Seasonal", ["additive", "multiplicative", "none"])
                        manual_params = {'trend': map_model_mode(t), 'seasonal': map_model_mode(s)}
                    elif "Theta" in sel_model:
                        manual_params['theta'] = st.number_input("Theta", 0.0, 5.0, 2.0)
                    elif "Prophet" in sel_model:
                        manual_params['sm'] = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
                        manual_params['cp'] = st.slider("Flexibility", 0.01, 0.5, 0.05)
                    elif "LightGBM" in sel_model or "Linear" in sel_model:
                        manual_params['lags'] = st.slider("Lags", 1, 60, 12)
                        if "LightGBM" in sel_model: manual_params['lr'] = st.number_input("LR", 0.001, 0.5, 0.1)
                    elif "ARIMA (Manual)" in sel_model:
                        c1, c2, c3 = st.columns(3)
                        p = c1.number_input("p", 0, 5, 1)
                        d = c2.number_input("d", 0, 2, 1)
                        q = c3.number_input("q", 0, 5, 1)
                        manual_params['order'] = (p, d, q)
                        manual_params['seas'] = (0, 0, 0, 0)
                        if st.checkbox("Seasonal?"):
                            manual_params['seas'] = (0, 1, 0, period_in)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 РАССЧИТАТЬ ПРОГНОЗ"):
            ts = TimeSeries.from_dataframe(df.reset_index(), time_col=d_col, value_cols=t_col)
            ts = fill_missing_values(ts)

            val_len = horizon if horizon < len(ts) * 0.3 else int(len(ts) * 0.2)
            train, val = ts.split_before(len(ts) - val_len)
            metric_func = {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape}[metric_in]

            status_man = st.status(f"Запуск {sel_model}...", expanded=True)
            m_obj = None

            # 1. AUTO
            if "AutoARIMA" in sel_model:
                if AUTOARIMA_AVAILABLE: m_obj = AutoARIMA(seasonal=True)

            # 2. OPTUNA
            elif use_optuna:
                status_man.write("Подбор параметров...")
                m_obj, bp = run_optimization(sel_model, train, val, metric_func, period_in, n_trials=n_trials_man)
                status_man.write(f"Лучшие: {bp}")

            # 3. MANUAL
            else:
                if "Exponential" in sel_model:
                    m_obj = ExponentialSmoothing(trend=manual_params['trend'], seasonal=manual_params['seasonal'],
                                                 seasonal_periods=period_in)
                elif "Theta" in sel_model:
                    m_obj = Theta(theta=manual_params['theta'])
                elif "Prophet" in sel_model:
                    m_obj = Prophet(seasonality_mode=manual_params['sm'], changepoint_prior_scale=manual_params['cp'])
                elif "LightGBM" in sel_model:
                    m_obj = LightGBMModel(lags=manual_params['lags'], learning_rate=manual_params['lr'],
                                          output_chunk_length=1)
                elif "Linear" in sel_model:
                    m_obj = LinearRegressionModel(lags=manual_params['lags'])
                elif "Manual" in sel_model:
                    m_obj = ARIMA(p=manual_params['order'][0], d=manual_params['order'][1], q=manual_params['order'][2],
                                  seasonal_order=manual_params['seas'])

            if m_obj:
                m_obj.fit(train)
                pv = m_obj.predict(len(val))

                status_man.update(label="Готово!", state="complete", expanded=False)

                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("MAE", f"{mae(val, pv):.2f}")
                c2.metric("RMSE", f"{rmse(val, pv):.2f}")
                c3.metric("MSE", f"{mse(val, pv):.2f}")
                c4.metric("MAPE", f"{mape(val, pv):.2f}%")

                # Forecast
                m_obj.fit(ts)
                pf = m_obj.predict(horizon)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=ts.time_index, y=ts.values().flatten(), name="История", line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=val.time_index, y=pv.values().flatten(), name="Валидация",
                                         line=dict(color='orange', dash='dot')))
                fig.add_trace(go.Scatter(x=pf.time_index, y=pf.values().flatten(), name="ПРОГНОЗ",
                                         line=dict(color='#00CC66', width=3)))
                st.plotly_chart(fig, use_container_width=True)

                # Export
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w:
                    safe_export_df(ts, 'Hist').to_excel(w, sheet_name='Hist', index=False)
                    safe_export_df(pf, 'Fcst').to_excel(w, sheet_name='Fcst', index=False)
                b.seek(0)
                st.download_button("📥 Скачать Excel", b, "forecast.xlsx")

else:
    st.info("👈 Загрузите файл в меню слева.")