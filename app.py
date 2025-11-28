
import streamlit as st
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mape, r2_score
import warnings
import matplotlib.pyplot as plt
from models_lib import MODELS, train_model
from utils import (
    plot_decomposition,
    plot_forecast,
    create_excel_download,
    export_fig_to_png,
)

# --- Page Config ---
st.set_page_config(
    page_title="Битва моделей временных рядов",
    page_icon="⚔️",
    layout="wide",
)

# --- Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Session State Management ---
def init_session_state():
    """Initializes session state variables if they don't exist."""
    defaults = {
        "screen": "upload",
        "df": None,
        "time_col": None,
        "value_col": None,
        "extra_cols": [],
        "n_forecast": 12,
        "season_period": 12,
        "ranking_metric": "MAPE",
        "battle_results": None,
        "trained_models": None,
        "forecasts": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session():
    """Resets the session state to start over."""
    st.session_state.clear()
    init_session_state()

# --- Main App Logic ---
init_session_state()

# --- SCREEN 1: UPLOAD ---
if st.session_state.screen == "upload":
    st.title("⚔️ Битва моделей временных рядов")
    st.header("Шаг 1: Загрузите ваш файл")

    uploaded_file = st.file_uploader(
        "Выберите CSV или Excel файл", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, parse_dates=True)
            else:
                df = pd.read_excel(uploaded_file)

            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                    except (ValueError, TypeError):
                        continue
            
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols:
                df.dropna(subset=date_cols, inplace=True)

            st.session_state.df = df
            st.session_state.screen = "setup"
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")

# --- SCREEN 2: SETUP ---
elif st.session_state.screen == "setup":
    st.title("Шаг 2: Анализ и настройка")
    df = st.session_state.df

    st.subheader("Предпросмотр данных")
    st.dataframe(df.head())

    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not date_cols:
        st.error("Не найдено колонок с датами. Пожалуйста, проверьте ваш файл.")
        if st.button("Начать заново"): reset_session(); st.rerun()
        st.stop()

    st.session_state.time_col = st.selectbox("1. Выберите колонку с датой/временем:", date_cols, index=date_cols.index(st.session_state.time_col) if st.session_state.time_col in date_cols else 0)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    available_value_cols = [col for col in numeric_cols if col != st.session_state.time_col]

    if not available_value_cols:
        st.error("В файле не найдено числовых колонок для анализа.")
        if st.button("Начать заново"): reset_session(); st.rerun()
        st.stop()

    st.session_state.value_col = st.selectbox("2. Выберите колонку со значениями:", available_value_cols, index=available_value_cols.index(st.session_state.value_col) if st.session_state.value_col in available_value_cols else 0)
    
    available_extra_cols = [col for col in numeric_cols if col not in [st.session_state.time_col, st.session_state.value_col]]
    st.session_state.extra_cols = st.multiselect("3. Выберите доп. факторы (необязательно):", available_extra_cols, default=st.session_state.extra_cols)

    st.subheader("Настройки анализа")
    
    df_for_series = df.sort_values(by=st.session_state.time_col).copy()
    df_for_series[st.session_state.value_col] = pd.to_numeric(df_for_series[st.session_state.value_col], errors='coerce')
    df_for_series.dropna(subset=[st.session_state.value_col], inplace=True)
    
    series = TimeSeries.from_dataframe(df_for_series, time_col=st.session_state.time_col, value_cols=st.session_state.value_col, fill_missing_dates=True, freq=None)
    
    inferred_freq = pd.infer_freq(series.time_index)
    if inferred_freq:
        st.info(f"Определенная частота временного ряда: {inferred_freq}")
        if 'D' in inferred_freq and st.session_state.get('freq_set_auto', False) is False:
            st.session_state.season_period = 7
            st.session_state.freq_set_auto = True
            st.rerun()
    else:
        st.warning("Не удалось автоматически определить частоту ряда.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.n_forecast = st.number_input("4. Укажите срок прогнозирования (в шагах):", min_value=1, value=st.session_state.n_forecast, step=1)
        st.session_state.season_period = st.number_input("5. Укажите период сезонности:", min_value=2, value=st.session_state.season_period, step=1)
        
        st.subheader("Метрика для ранжирования")
        if 0 in series.values():
            st.session_state.ranking_metric = "MAE"
            st.info("Рекомендуемая метрика: **MAE**")
            st.markdown("В ваших данных присутствуют нулевые значения, поэтому **MAPE** (процентная ошибка) не может быть использована. **MAE** (средняя абсолютная ошибка) является лучшей альтернативой.")
        else:
            st.session_state.ranking_metric = "MAPE"
            st.info("Рекомендуемая метрика: **MAPE**")
            st.markdown("**MAPE** (средняя абсолютная процентная ошибка) отлично подходит для ваших данных, так как показывает ошибку в процентах.")

    with col2:
        st.subheader("Анализ сезонности")
        # FIX: Add a strict check to prevent plotting if data is insufficient
        if len(series) < 2 * st.session_state.season_period:
            st.warning(f"Недостаточно данных для анализа сезонности. Требуется как минимум {2 * st.session_state.season_period} точек данных (2 периода), а у вас {len(series)}. График не будет построен.")
        else:
            try:
                plt.figure(figsize=(10, 6))
                plot_decomposition(series, st.session_state.value_col, period=st.session_state.season_period)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.warning(f"Не удалось построить график декомпозиции: {e}")

    if st.button("🚀 Начать битву моделей!", type="primary"):
        st.session_state.screen = "results"
        st.rerun()

# --- SCREEN 3: RESULTS ---
elif st.session_state.screen == "results":
    st.title("Шаг 3: Результаты битвы")

    if st.button("↩️ Начать заново"):
        reset_session()
        st.rerun()

    if st.session_state.battle_results is None:
        df = st.session_state.df
        time_col = st.session_state.time_col
        value_col = st.session_state.value_col
        n_forecast = st.session_state.n_forecast
        extra_cols = st.session_state.extra_cols

        try:
            df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
            cols_to_process = [value_col] + extra_cols
            for col in cols_to_process:
                df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')
            df_sorted.dropna(subset=cols_to_process, inplace=True)

            series = TimeSeries.from_dataframe(df_sorted, time_col, value_col, fill_missing_dates=True, freq=None).astype(np.float32)
            
            # FIX: Add a strict check for minimum training size to prevent IndexError
            min_train_size = 10 
            if (len(series) - n_forecast) < min_train_size:
                st.error(f"Ошибка: Недостаточно данных для обучения. Требуется как минимум {min_train_size} точек данных для обучения, но после выделения горизонта прогноза ({n_forecast}) остается только {len(series) - n_forecast}. Пожалуйста, уменьшите срок прогноза или загрузите больше данных.")
                st.stop()

            train, val = series[:-n_forecast], series[-n_forecast:]
            
            future_covariates = None
            if extra_cols:
                future_covariates = TimeSeries.from_dataframe(df_sorted, time_col, extra_cols, fill_missing_dates=True, freq=None).astype(np.float32)

            models_to_run = {name: mi for name, mi in MODELS.items() if not (mi["requires_extras"] and not extra_cols)}
            
            results_list, forecasts, trained_models = [], {}, {}
            progress_bar = st.progress(0, text="Начинаем битву...")
            
            for i, (name, model_info) in enumerate(models_to_run.items()):
                progress_bar.progress((i + 1) / len(models_to_run), text=f"Обучается: {name}")
                
                forecast, model, error = train_model(
                    model_name=name, train_series=train,
                    forecast_horizon=len(val), future_covariates=future_covariates
                )

                if error or forecast is None:
                    results_list.append({"Модель": name, "MAPE": np.nan, "MAE": np.nan, "R2": np.nan, "Гиперпараметры": error or "Неизвестная ошибка"})
                    continue

                mape_score = mape(val, forecast) if 0 not in val.values() else np.nan
                mae_score = mae(val, forecast)
                r2_score_val = r2_score(val, forecast)
                
                params = model.model_params if hasattr(model, 'model_params') else model.get_params()
                
                results_list.append({"Модель": name, "MAPE": mape_score, "MAE": mae_score, "R2": r2_score_val, "Гиперпараметры": str(params)})
                forecasts[name] = forecast
                trained_models[name] = model
            
            progress_bar.empty()

            if not results_list:
                st.error("Ни одна модель не смогла быть обучена."); st.stop()

            results_df = pd.DataFrame(results_list).set_index("Модель")
            results_df = results_df.sort_values(by=st.session_state.ranking_metric, ascending=st.session_state.ranking_metric != "R2", na_position='last')
            
            st.session_state.battle_results = results_df
            st.session_state.forecasts = forecasts
            st.session_state.trained_models = trained_models

        except Exception as e:
            st.error(f"Произошла критическая ошибка на этапе выполнения: {e}")
            st.exception(e)
            st.stop()

    results_df = st.session_state.battle_results
    forecasts = st.session_state.forecasts
    
    st.subheader(f"🏆 Таблица результатов (прогноз на {st.session_state.n_forecast} шагов)")
    st.markdown(f"Ранжирование по: **{st.session_state.ranking_metric}**")

    def highlight_best(s):
        is_min = s.name in ["MAE", "MAPE"]
        best_val = s.min() if is_min else s.max()
        return ['background-color: #28a745' if v == best_val else '' for v in s]

    st.dataframe(results_df.style.apply(highlight_best, subset=["MAE", "MAPE", "R2"]).format({"MAPE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}"}, na_rep="-"))

    st.subheader("📊 График прогнозов")
    successful_models = list(forecasts.keys())
    if successful_models:
        default_models = results_df.dropna(subset=[st.session_state.ranking_metric]).head(3).index.tolist()
        
        models_to_plot = st.multiselect("Выберите модели для отображения:", successful_models, default=default_models)
        
        if models_to_plot:
            series_to_plot = TimeSeries.from_dataframe(st.session_state.df, st.session_state.time_col, st.session_state.value_col, fill_missing_dates=True, freq=None).astype(np.float32)
            train_plot, val_plot = series_to_plot[:-st.session_state.n_forecast], series_to_plot[-st.session_state.n_forecast:]
            
            selected_forecasts = {name: forecasts[name] for name in models_to_plot if name in forecasts}
            fig = plot_forecast(train_plot, val_plot, selected_forecasts)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📥 Выгрузка результатов")
            best_model_name = results_df.dropna(subset=[st.session_state.ranking_metric]).index[0]
            if best_model_name in forecasts:
                best_forecast = forecasts[best_model_name]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # FIX: Correct method call from .pd_dataframe to .pd_dataframe()
                    st.download_button("Скачать CSV (прогноз)", best_forecast.pd_dataframe().to_csv(index=True).encode("utf-8"), f"forecast_{best_model_name}.csv", "text/csv")
                with col2:
                    # FIX: Pass a dataframe to the excel function
                    st.download_button("Скачать XLSX (прогноз)", create_excel_download(best_forecast.pd_dataframe()), f"forecast_{best_model_name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col3:
                    st.download_button("Скачать PNG (график)", export_fig_to_png(fig), "forecast_plot.png", "image/png")
            else:
                st.warning("Лучшая модель по метрике не смогла быть построена, выгрузка невозможна.")
    else:
        st.warning("Ни одна модель не смогла построить прогноз, график недоступен.")
