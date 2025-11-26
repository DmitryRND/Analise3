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

st.set_page_config(page_title="TS Master v9.0", layout="wide")


# --- HELPERS ---

def get_safe_lags(data_len):
    max_lags = int(data_len / 2) - 1
    return min(12, max_lags) if max_lags > 1 else 1


def map_model_mode(val_str):
    if val_str == "additive": return ModelMode.ADDITIVE
    if val_str == "multiplicative": return ModelMode.MULTIPLICATIVE
    if val_str == "none": return ModelMode.NONE
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


def safe_export_df(ts_obj, col_name='Value'):
    try:
        vals = ts_obj.values().flatten()
        idx = ts_obj.time_index
        return pd.DataFrame(data=vals, index=idx, columns=[col_name]).reset_index()
    except:
        return pd.DataFrame(columns=['Error'])


# --- OPTUNA ENGINE ---

def run_optimization(model_name, train, val, metric_func, period):
    def objective(trial):
        try:
            m_tmp = None
            if model_name == "ExponentialSmoothing":
                tr_str = trial.suggest_categorical("trend", ["additive", "multiplicative", "none"])
                se_str = trial.suggest_categorical("seasonal", ["additive", "multiplicative", "none"])
                m_tmp = ExponentialSmoothing(
                    trend=map_model_mode(tr_str),
                    seasonal=map_model_mode(se_str),
                    seasonal_periods=period
                )
            elif model_name == "Theta":
                th = trial.suggest_float("theta", 0.1, 5.0)
                mod_str = trial.suggest_categorical("mode", ["additive", "multiplicative"])
                m_tmp = Theta(theta=th, season_mode=map_seasonality_mode(mod_str))
            elif model_name == "LightGBM":
                l = trial.suggest_int("lags", 4, 30)
                lr = trial.suggest_float("learning_rate", 0.01, 0.3)
                m_tmp = LightGBMModel(lags=l, learning_rate=lr, output_chunk_length=1, verbose=-1)
            elif model_name == "LinearRegression":
                l = trial.suggest_int("lags", 1, 40)
                m_tmp = LinearRegressionModel(lags=l)
            elif model_name == "Prophet":
                sm = trial.suggest_categorical("sm", ["additive", "multiplicative"])
                cp = trial.suggest_float("cp", 0.01, 0.5)
                m_tmp = Prophet(seasonality_mode=sm, changepoint_prior_scale=cp)

            m_tmp.fit(train)
            p = m_tmp.predict(len(val))
            return metric_func(val, p)
        except Exception:
            return float('inf')

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    bp = study.best_params

    final_model = None
    if model_name == "ExponentialSmoothing":
        final_model = ExponentialSmoothing(
            trend=map_model_mode(bp['trend']),
            seasonal=map_model_mode(bp['seasonal']),
            seasonal_periods=period
        )
    elif model_name == "Theta":
        final_model = Theta(theta=bp['theta'], season_mode=map_seasonality_mode(bp['mode']))
    elif model_name == "LightGBM":
        final_model = LightGBMModel(lags=bp['lags'], learning_rate=bp['learning_rate'], output_chunk_length=1)
    elif model_name == "LinearRegression":
        final_model = LinearRegressionModel(lags=bp['lags'])
    elif model_name == "Prophet":
        final_model = Prophet(seasonality_mode=bp['sm'], changepoint_prior_scale=bp['cp'])

    return final_model, bp


# --- UI ---

st.title("🛡️ TS Master v9.0 (Fixed)")

# 1. DATA
st.sidebar.header("1. Данные")
uploaded_file = st.sidebar.file_uploader("CSV / XLSX", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    cols = df.columns.tolist()
    date_guess = next((c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'дата' in c.lower()), cols[0])
    target_guess = next((c for c in cols if c != date_guess and pd.api.types.is_numeric_dtype(df[c])),
                        cols[1] if len(cols) > 1 else cols[0])

    c1, c2 = st.sidebar.columns(2)
    date_col = c1.selectbox("Дата", cols, index=cols.index(date_guess))
    target_col = c2.selectbox("Значение", cols, index=cols.index(target_guess))

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).set_index(date_col)
        df[target_col] = df[target_col].interpolate()
    except Exception as e:
        st.error(e)
        st.stop()

    # 2. PARAMS
    st.header("2. Параметры")

    col_h, col_p = st.columns(2)
    horizon = col_h.number_input("📅 Горизонт прогноза (шагов)", 1, 1000, 12)

    freq_detected = pd.infer_freq(df.index)
    default_period = 12
    if freq_detected and 'D' in freq_detected: default_period = 7
    period_input = col_p.number_input("🔄 Период сезонности", 2, 365, default_period)

    has_seas, decomp = check_seasonality(df.reset_index(), target_col, period_input)
    if decomp:
        with st.expander("Показать график сезонности"):
            fig_diag = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig_diag.add_trace(go.Scatter(x=df.index, y=df[target_col], name='Fact'), row=1, col=1)
            fig_diag.add_trace(go.Scatter(x=df.index, y=decomp.seasonal, name='Seasonality', line=dict(color='green')),
                               row=2, col=1)
            st.plotly_chart(fig_diag, width='stretch')

    # 3. MODEL SELECTION
    st.markdown("---")
    c_algo, c_mode, c_met = st.columns(3)

    opts = ["🏆 БИТВА МОДЕЛЕЙ", "ExponentialSmoothing", "ARIMA (Manual)", "Theta", "LinearRegression", "LightGBM",
            "Prophet"]
    if AUTOARIMA_AVAILABLE: opts.insert(2, "AutoARIMA")

    model_name = c_algo.selectbox("Алгоритм", opts)

    # Flags
    is_battle = "БИТВА" in model_name
    is_manual_arima = "ARIMA (Manual)" in model_name

    use_deep_battle = False
    is_manual_mode = False  # Флаг для отображения ручных настроек

    if is_battle:
        use_deep_battle = c_mode.checkbox("Глубокая битва (Optuna)", value=False)
    elif is_manual_arima:
        is_manual_mode = True
        c_mode.info("Ручной режим (Manual)")
    else:
        mode_select = c_mode.radio("Режим", ["Ручной (Manual)", "AutoML (Optuna)"])
        if mode_select == "Ручной (Manual)":
            is_manual_mode = True

    metric_name = c_met.selectbox("Цель", ["MAE", "RMSE", "MSE", "MAPE"])
    METRICS = {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape}
    metric_func = METRICS[metric_name]

    # MANUAL PARAMS UI
    params = {}
    if is_manual_mode and not is_battle:
        with st.expander("🎛 Настройки вручную", expanded=True):
            if "ARIMA (Manual)" in model_name:
                c1, c2, c3 = st.columns(3)
                p = c1.number_input("p", 0, 10, 1)
                d = c2.number_input("d", 0, 2, 1)
                q = c3.number_input("q", 0, 10, 1)
                is_seas = st.checkbox("Seasonal?", value=True)
                params['order'] = (p, d, q)
                params['seas_order'] = (0, 1, 0, period_input) if is_seas else (0, 0, 0, 0)
            elif "LightGBM" in model_name:
                params['lags'] = st.slider("Lags", 1, 60, 12)
                params['lr'] = st.number_input("LR", 0.01, 0.5, 0.1)
            elif "LinearRegression" in model_name:
                params['lags'] = st.slider("Lags", 1, 60, 12)
            elif "Theta" in model_name:
                params['theta'] = st.number_input("Theta", 0.0, 5.0, 2.0)
            elif "Prophet" in model_name:
                params['mode'] = st.selectbox("Mode", ["additive", "multiplicative"])
                params['cps'] = st.slider("Flexibility", 0.01, 0.5, 0.05)
            elif "ExponentialSmoothing" in model_name:
                c1, c2 = st.columns(2)
                t_str = c1.selectbox("Trend", ["additive", "multiplicative", "none"])
                s_str = c2.selectbox("Seasonal", ["additive", "multiplicative", "none"])
                params['trend'] = map_model_mode(t_str)
                params['seasonal'] = map_model_mode(s_str)

    # --- EXECUTION ---
    if st.button("🚀 ЗАПУСК"):
        ts = TimeSeries.from_dataframe(df.reset_index(), time_col=date_col, value_cols=target_col)
        ts = fill_missing_values(ts)

        # Split
        val_len = horizon if horizon < len(ts) * 0.3 else int(len(ts) * 0.2)
        train, val = ts.split_before(len(ts) - val_len)
        safe_lags = get_safe_lags(len(train))

        # === BATTLE MODE ===
        if is_battle:
            st.subheader("🥊 Арена Битвы")

            # Models List
            models_list = [
                ("ExponentialSmoothing", ExponentialSmoothing(seasonal_periods=period_input)),
                ("Theta", Theta()),
                ("LinearRegression", LinearRegressionModel(lags=safe_lags)),
                ("LightGBM", LightGBMModel(lags=safe_lags, output_chunk_length=1, verbose=-1)),
                ("Prophet", Prophet())
            ]

            if AUTOARIMA_AVAILABLE:
                # FIX: Removed suppress_warnings argument
                try:
                    aa = AutoARIMA(seasonal=True)
                    models_list.append(("AutoARIMA", aa))
                except:
                    pass

            results = []
            progress_bar = st.progress(0)
            status = st.empty()

            for i, (m_name, default_model) in enumerate(models_list):
                status.text(f"Боец: {m_name}...")
                try:
                    final_m = default_model
                    p_info = "Default"

                    if use_deep_battle and m_name != "AutoARIMA":
                        final_m, best_p = run_optimization(m_name, train, val, metric_func, period_input)
                        p_info = str(best_p)

                    final_m.fit(train)
                    pred = final_m.predict(len(val))

                    # Calculate ALL metrics
                    s_mae = mae(val, pred)
                    s_rmse = rmse(val, pred)
                    s_mse = mse(val, pred)
                    s_mape = mape(val, pred)

                    # Target metric for sorting
                    target_score = metric_func(val, pred)

                    results.append({
                        "Model": m_name,
                        "Score": target_score,  # For sorting
                        "MAE": s_mae, "RMSE": s_rmse, "MSE": s_mse, "MAPE": s_mape,
                        "Obj": final_m, "Pred": pred, "Params": p_info
                    })
                except Exception as e:
                    pass
                progress_bar.progress((i + 1) / len(models_list))

            status.text("Готово!")

            if results:
                # Sort
                res_df = pd.DataFrame(results).sort_values("Score")

                # Leaderboard
                st.dataframe(res_df[["Model", "MAE", "RMSE", "MSE", "MAPE", "Params"]].style.highlight_min(axis=0,
                                                                                                           subset=[
                                                                                                               "MAE",
                                                                                                               "RMSE",
                                                                                                               "MSE",
                                                                                                               "MAPE"],
                                                                                                           color='lightgreen'))

                best = res_df.iloc[0]
                st.success(f"🏆 Победитель по {metric_name}: **{best['Model']}**")

                # Winner Metrics Display
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{best['MAE']:.2f}")
                m2.metric("RMSE", f"{best['RMSE']:.2f}")
                m3.metric("MSE", f"{best['MSE']:.2f}")
                m4.metric("MAPE", f"{best['MAPE']:.2f}%")

                # Battle Plot
                fig_b = go.Figure()
                fig_b.add_trace(go.Scatter(x=val.time_index, y=val.values().flatten(), name="FACT",
                                           line=dict(color='black', width=3)))
                for res in results:
                    is_best = (res['Model'] == best['Model'])
                    op = 1.0 if is_best else 0.3
                    width = 4 if is_best else 1
                    fig_b.add_trace(
                        go.Scatter(x=res['Pred'].time_index, y=res['Pred'].values().flatten(), name=res['Model'],
                                   opacity=op, line=dict(width=width)))
                st.plotly_chart(fig_b, width='stretch')

                # FINAL FORECAST (REFIT)
                st.subheader("🔮 Прогноз в будущее (Best Model)")
                with st.spinner("Переобучение победителя..."):
                    best_obj = best['Obj']
                    best_obj.fit(ts)  # Refit on full data
                    final_fcst = best_obj.predict(horizon)

                # Final Plot
                fig_f = go.Figure()
                fig_f.add_trace(
                    go.Scatter(x=ts.time_index, y=ts.values().flatten(), name="History", line=dict(color='gray')))
                fig_f.add_trace(go.Scatter(x=final_fcst.time_index, y=final_fcst.values().flatten(), name="FORECAST",
                                           line=dict(color='green', width=3)))
                st.plotly_chart(fig_f, width='stretch')

                # Export
                b = BytesIO()
                with pd.ExcelWriter(b, engine='openpyxl') as w:
                    safe_export_df(ts, 'Hist').to_excel(w, sheet_name='Hist', index=False)
                    safe_export_df(final_fcst, 'Fcst').to_excel(w, sheet_name='Fcst', index=False)
                    res_df[["Model", "MAE", "RMSE", "MSE", "MAPE", "Params"]].to_excel(w, sheet_name='Leaderboard',
                                                                                       index=False)
                b.seek(0)
                st.download_button("📥 Скачать результаты Битвы (XLSX)", b, "battle_results.xlsx")

        # === SINGLE MODE ===
        else:
            m_obj = None
            with st.spinner("Работаем..."):

                # 1. AUTO ARIMA
                if "AutoARIMA" in model_name:
                    if AUTOARIMA_AVAILABLE:
                        # FIX: Remove suppress_warnings
                        m_obj = AutoARIMA(seasonal=True)
                    else:
                        st.error("No pmdarima")

                # 2. OPTUNA
                elif not is_manual_mode:
                    m_obj, bp = run_optimization(model_name, train, val, metric_func, period_input)
                    st.success(f"Optuna Best: {bp}")

                # 3. MANUAL
                else:
                    if "ARIMA (Manual)" in model_name:
                        # FIX: Explicit params access
                        m_obj = ARIMA(p=params['order'][0], d=params['order'][1], q=params['order'][2],
                                      seasonal_order=params['seas_order'])
                    elif "LightGBM" in model_name:
                        m_obj = LightGBMModel(lags=params['lags'], learning_rate=params['lr'], output_chunk_length=1)
                    elif "LinearRegression" in model_name:
                        m_obj = LinearRegressionModel(lags=params['lags'])
                    elif "Theta" in model_name:
                        m_obj = Theta(theta=params['theta'])
                    elif "Prophet" in model_name:
                        m_obj = Prophet(seasonality_mode=params['mode'], changepoint_prior_scale=params['cps'])
                    elif "ExponentialSmoothing" in model_name:
                        m_obj = ExponentialSmoothing(trend=params['trend'], seasonal=params['seasonal'],
                                                     seasonal_periods=period_input)

                if m_obj:
                    # Train & Val
                    m_obj.fit(train)
                    pv = m_obj.predict(len(val))

                    # Metrics
                    s_mae = mae(val, pv)
                    s_rmse = rmse(val, pv)
                    s_mse = mse(val, pv)
                    s_mape = mape(val, pv)

                    # Display Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MAE", f"{s_mae:.2f}")
                    m2.metric("RMSE", f"{s_rmse:.2f}")
                    m3.metric("MSE", f"{s_mse:.2f}")
                    m4.metric("MAPE", f"{s_mape:.2f}%")

                    # Full Forecast
                    m_obj.fit(ts)
                    pf = m_obj.predict(horizon)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(x=ts.time_index, y=ts.values().flatten(), name="Hist", line=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=val.time_index, y=pv.values().flatten(), name="Val",
                                             line=dict(color='orange', dash='dot')))
                    fig.add_trace(go.Scatter(x=pf.time_index, y=pf.values().flatten(), name="Fcst",
                                             line=dict(color='green', width=3)))
                    st.plotly_chart(fig, width='stretch')

                    # Export
                    b = BytesIO()
                    with pd.ExcelWriter(b, engine='openpyxl') as w:
                        safe_export_df(ts, 'Hist').to_excel(w, sheet_name='Hist', index=False)
                        safe_export_df(pf, 'Fcst').to_excel(w, sheet_name='Fcst', index=False)
                    b.seek(0)
                    st.download_button("📥 Скачать Excel", b, "fcst.xlsx")

else:
    st.info("Загрузите CSV или Excel файл.")