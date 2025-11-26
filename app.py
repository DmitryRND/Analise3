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
st.set_page_config(page_title="TS Master v13.0", page_icon="🧠", layout="wide")

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


def perform_decomposition(df, value_col, period):
    try:
        # Проверка на достаточное количество данных
        if len(df) < period * 2: return None

        decomposition = seasonal_decompose(df[value_col], model='additive', period=int(period))
        return decomposition
    except:
        return None


def detect_outliers_zscore(df, value_col):
    if len(df) < 10: return False
    z = np.abs(stats.zscore(df[value_col]))
    return np.any(z > 3)


def safe_export_df(ts_obj, col_name='Value'):
    try:
        vals = ts_obj.values().flatten()
        idx = ts_obj.time_index
        return pd.DataFrame(data=vals, index=idx, columns=[col_name]).reset_index()
    except:
        return pd.DataFrame(columns=['Error'])


# --- OPTUNA CORE ---

def run_optimization(model_name, train, val, metric_func, period, past_covariates=None, n_trials=10):
    # Флаг: поддерживает ли модель ковариаты
    supports_covariates = model_name in ["LightGBM", "LinearRegression"] and past_covariates is not None

    def objective(trial):
        try:
            m_tmp = None
            # --- MODEL DEFINITIONS ---
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
                # Если есть ковариаты, добавляем lags_past_covariates
                l_cov = [l] if supports_covariates else None
                m_tmp = LightGBMModel(lags=l, lags_past_covariates=l_cov, learning_rate=lr, output_chunk_length=1,
                                      verbose=-1)

            elif model_name == "Prophet":
                sm = trial.suggest_categorical("sm", ["additive", "multiplicative"])
                cp = trial.suggest_float("cp", 0.01, 0.5)
                m_tmp = Prophet(seasonality_mode=sm, changepoint_prior_scale=cp)

            elif model_name == "LinearRegression":
                l = trial.suggest_int("lags", 4, 40)
                l_cov = [l] if supports_covariates else None
                m_tmp = LinearRegressionModel(lags=l, lags_past_covariates=l_cov)

            # --- FIT & PREDICT ---
            if supports_covariates:
                m_tmp.fit(train, past_covariates=past_covariates)
                p = m_tmp.predict(len(val), past_covariates=past_covariates)
            else:
                m_tmp.fit(train)
                p = m_tmp.predict(len(val))

            return metric_func(val, p)
        except Exception:
            return float('inf')

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    bp = study.best_params

    # --- REBUILD BEST MODEL ---
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
        l_cov = [bp['lags']] if supports_covariates else None
        final_m = LightGBMModel(lags=bp['lags'], lags_past_covariates=l_cov, learning_rate=bp['lr'],
                                output_chunk_length=1)
    elif model_name == "Prophet":
        final_m = Prophet(seasonality_mode=bp['sm'], changepoint_prior_scale=bp['cp'])
    elif model_name == "LinearRegression":
        l_cov = [bp['lags']] if supports_covariates else None
        final_m = LinearRegressionModel(lags=bp['lags'], lags_past_covariates=l_cov)

    return final_m, bp


# --- UI START ---

st.title("🛡️ TS Master v13.0 (Multivariate Support)")

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
        t_col = st.selectbox("Целевая переменная (Target)", [c for c in cols if c != d_col], index=0)

        # EXOGENOUS SELECTION
        possible_covs = [c for c in cols if c not in [d_col, t_col] and pd.api.types.is_numeric_dtype(df[c])]
        cov_cols = []
        if possible_covs:
            st.markdown("---")
            st.caption("Экзогенные переменные (Факторы влияния)")
            cov_cols = st.multiselect("Выберите доп. факторы", possible_covs)
            if cov_cols:
                st.info(
                    f"Выбрано факторов: {len(cov_cols)}. Они будут использованы в моделях LightGBM и LinearRegression.")

if df is not None:
    # --- PREPROCESSING START ---
    try:
        # 1. Date Parsing with dayfirst=True for DD.MM.YYYY support
        df[d_col] = pd.to_datetime(df[d_col], dayfirst=True, errors='coerce')

        # 2. Sort
        df = df.sort_values(by=d_col)

        # 3. Clean Tail (Drop rows where Target is NaN - common in templates)
        df_clean = df.dropna(subset=[t_col]).copy()

        # 4. Set Index & Interpolate
        df_clean = df_clean.set_index(d_col)
        df_clean[t_col] = df_clean[t_col].interpolate()

        # 5. Handle Covariates (Interpolate them too)
        for cc in cov_cols:
            df_clean[cc] = df_clean[cc].interpolate()

        # 6. Infer Frequency explicitly
        if pd.infer_freq(df_clean.index):
            df_clean = df_clean.asfreq(pd.infer_freq(df_clean.index))

    except Exception as e:
        st.error(f"Ошибка предобработки данных: {e}")
        st.stop()

    # Stats
    freq_detected = pd.infer_freq(df_clean.index)
    rec_period = 12
    if freq_detected and ('D' in freq_detected or 'W' in freq_detected): rec_period = 7
    if freq_detected and 'H' in freq_detected: rec_period = 24

    has_outliers = detect_outliers_zscore(df_clean, t_col)

    # 2. SETTINGS & DECOMPOSITION
    st.header("2. Анализ и Настройки")

    rec_metric = "MAE" if has_outliers else "RMSE"

    # Info Block
    st.markdown(f"""
    <div class="info-box">
        <b>💡 Данные загружены:</b><br>
        • Записей: <b>{len(df_clean)}</b><br>
        • Частота: <b>{freq_detected if freq_detected else "Не определена (используем RangeIndex)"}</b><br>
        • Доп. факторы: <b>{", ".join(cov_cols) if cov_cols else "Нет"}</b>
    </div>
    """, unsafe_allow_html=True)

    # Settings
    c1, c2, c3 = st.columns(3)
    horizon = c1.number_input("📅 Горизонт прогноза", 1, 1000, 12)
    period_in = c2.number_input("🔄 Период сезонности", 2, 365, rec_period)
    metric_in = c3.selectbox("🎯 Метрика", ["MAE", "RMSE", "MSE", "MAPE"],
                             index=["MAE", "RMSE", "MSE", "MAPE"].index(rec_metric))

    # --- DECOMPOSITION PLOT ---
    decomp = perform_decomposition(df_clean, t_col, period_in)
    if decomp:
        with st.expander("📊 Декомпозиция ряда (Тренд / Сезонность / Шум)", expanded=True):
            fig_dec = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    subplot_titles=("Тренд (Направление)", "Сезонность (Циклы)", "Остатки (Шум)"))
            fig_dec.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name='Trend', line=dict(color='orange')),
                              row=1, col=1)
            fig_dec.add_trace(
                go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name='Seasonal', line=dict(color='green')),
                row=2, col=1)
            fig_dec.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name='Resid', mode='markers',
                                         marker=dict(color='gray', size=3)), row=3, col=1)
            fig_dec.update_layout(height=600)
            st.plotly_chart(fig_dec, use_container_width=True)

    # --- 3. EXECUTION MODE ---
    st.markdown("---")
    st.subheader("3. Запуск прогнозирования")

    mode = st.radio("Mode", ["🥊 Битва моделей", "🛠 Ручной выбор"], label_visibility="collapsed", horizontal=True)

    # PREPARE DARTS SERIES
    try:
        ts = TimeSeries.from_dataframe(df_clean.reset_index(), time_col=d_col, value_cols=t_col)
        ts = fill_missing_values(ts)

        # COVARIATES SERIES
        cov_ts = None
        if cov_cols:
            cov_ts = TimeSeries.from_dataframe(df_clean.reset_index(), time_col=d_col, value_cols=cov_cols)
            cov_ts = fill_missing_values(cov_ts)

    except Exception as e:
        st.error(f"Ошибка создания TimeSeries: {e}")
        st.stop()

    # Split Data
    val_len = horizon if horizon < len(ts) * 0.3 else int(len(ts) * 0.2)
    train, val = ts.split_before(len(ts) - val_len)

    # Split Covariates (Aligned with target)
    cov_train, cov_val = (None, None)
    if cov_ts:
        cov_train, cov_val = cov_ts.split_before(len(ts) - val_len)

    metric_func = {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape}[metric_in]

    # ---------------- BATTLE ----------------
    if "Битва" in mode:
        bc1, bc2 = st.columns([1, 3])
        n_trials_battle = bc1.select_slider("Точность (попыток Optuna)", options=[5, 10, 20], value=10)

        if st.button("🥊 НАЧАТЬ БИТВУ"):
            status = st.status("Идет соревнование...", expanded=True)

            # Fighters List
            fighters = [
                ("ExponentialSmoothing", ExponentialSmoothing(seasonal_periods=period_in)),
                ("Theta", Theta()),
                ("LinearRegression", LinearRegressionModel(lags=get_safe_lags(len(train)))),
                ("LightGBM", LightGBMModel(lags=get_safe_lags(len(train)), output_chunk_length=1, verbose=-1)),
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

                    # Check if model supports covariates
                    use_covs = (cov_ts is not None) and (m_name in ["LightGBM", "LinearRegression"])

                    # Optimize
                    if m_name != "AutoARIMA":
                        # Если модель поддерживает ковариаты, передаем их в оптимизатор
                        opt_covs = cov_ts if use_covs else None
                        final_m, bp = run_optimization(m_name, train, val, metric_func, period_in,
                                                       past_covariates=opt_covs, n_trials=n_trials_battle)
                        params_txt = str(bp)
                        if use_covs: params_txt += " + Exog"

                    # Train & Predict
                    if use_covs:
                        final_m.fit(train, past_covariates=cov_ts)  # Используем весь ряд ковариат как "прошлое"
                        pred = final_m.predict(len(val), past_covariates=cov_ts)
                    else:
                        final_m.fit(train)
                        pred = final_m.predict(len(val))

                    score = metric_func(val, pred)

                    results.append({
                        "Model": m_name, "Score": score,
                        "Obj": final_m, "Pred": pred, "Params": params_txt,
                        "MAE": mae(val, pred), "MAPE": mape(val, pred),
                        "UsesCov": use_covs
                    })
                except Exception as e:
                    # status.write(f"Error {m_name}: {e}")
                    pass

            status.update(label="Готово!", state="complete", expanded=False)

            if results:
                res_df = pd.DataFrame(results).sort_values("Score")
                best = res_df.iloc[0]

                st.success(f"🏆 Победитель: **{best['Model']}**")
                st.dataframe(res_df[["Model", "Score", "Params", "UsesCov"]].style.highlight_min(subset=["Score"],
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

                # Refit & Forecast
                # NOTE: For multivariate models, we need future values of covariates if future_covariates,
                # BUT for LightGBM/LinReg in Darts using 'past_covariates', we assume we only know past.
                # However, usually we want to use the known future values if available.
                # Since we stripped the tail, we don't have future values. We rely on the model purely.

                with st.spinner("Финальный прогноз..."):
                    best_obj = best['Obj']

                    if best['UsesCov']:
                        best_obj.fit(ts, past_covariates=cov_ts)
                        fcst = best_obj.predict(horizon, past_covariates=cov_ts)
                    else:
                        best_obj.fit(ts)
                        fcst = best_obj.predict(horizon)

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

    # ---------------- MANUAL ----------------
    else:
        m_opts = ["ARIMA (Optuna)", "ExponentialSmoothing", "LightGBM", "Prophet", "Theta", "LinearRegression",
                  "ARIMA (Manual)"]
        if AUTOARIMA_AVAILABLE: m_opts.insert(1, "AutoARIMA (Classic)")

        sel_model = st.selectbox("Модель", m_opts)

        # Flags
        can_use_covs = (cov_ts is not None) and (sel_model in ["LightGBM", "LinearRegression"])
        if can_use_covs:
            st.success(f"✅ Модель {sel_model} будет использовать экзогенные переменные: {cov_cols}")

        use_optuna = st.checkbox("Включить Optuna (Авто-подбор)", value=False)
        n_trials_man = 10
        if use_optuna:
            n_trials_man = st.slider("Попыток", 10, 50, 20)

        if st.button("🚀 РАССЧИТАТЬ"):
            status_man = st.status(f"Запуск {sel_model}...", expanded=True)
            m_obj = None

            # Init Model
            if "AutoARIMA" in sel_model:
                if AUTOARIMA_AVAILABLE: m_obj = AutoARIMA(seasonal=True)

            elif use_optuna or "Optuna" in sel_model:
                opt_covs = cov_ts if can_use_covs else None
                m_obj, bp = run_optimization(sel_model, train, val, metric_func, period_in, past_covariates=opt_covs,
                                             n_trials=n_trials_man)
                status_man.write(f"Параметры: {bp}")

            else:
                # Default Manual (Simplified for brevity, usually you'd add manual controls here like in v12)
                # Using smart defaults here
                safe_lags = get_safe_lags(len(train))
                if "Exponential" in sel_model:
                    m_obj = ExponentialSmoothing(seasonal_periods=period_in)
                elif "Theta" in sel_model:
                    m_obj = Theta()
                elif "Prophet" in sel_model:
                    m_obj = Prophet()
                elif "LightGBM" in sel_model:
                    m_obj = LightGBMModel(lags=safe_lags, output_chunk_length=1)
                elif "Linear" in sel_model:
                    m_obj = LinearRegressionModel(lags=safe_lags)
                elif "Manual" in sel_model:
                    m_obj = ARIMA(p=1, d=1, q=1)  # Basic default

            if m_obj:
                if can_use_covs:
                    m_obj.fit(train, past_covariates=cov_ts)
                    pv = m_obj.predict(len(val), past_covariates=cov_ts)
                else:
                    m_obj.fit(train)
                    pv = m_obj.predict(len(val))

                status_man.update(label="Готово!", state="complete", expanded=False)

                c1, c2 = st.columns(2)
                c1.metric("MAE", f"{mae(val, pv):.2f}")
                c2.metric("MAPE", f"{mape(val, pv):.2f}%")

                # Forecast
                if can_use_covs:
                    m_obj.fit(ts, past_covariates=cov_ts)
                    pf = m_obj.predict(horizon, past_covariates=cov_ts)
                else:
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

else:
    st.info("👈 Загрузите файл в меню слева.")