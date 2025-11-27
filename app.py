import streamlit as st
import pandas as pd
import numpy as np
from darts import TimeSeries
import warnings
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
warnings.filterwarnings("ignore")

# --- Session State ---
def init_session_state():
    if "screen" not in st.session_state:
        st.session_state.screen = "upload"
    if "df" not in st.session_state:
        st.session_state.df = None
    if "settings" not in st.session_state:
        st.session_state.settings = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "forecasts" not in st.session_state:
        st.session_state.forecasts = None
    if "time_col" not in st.session_state:
        st.session_state.time_col = None
    if "value_col" not in st.session_state:
        st.session_state.value_col = None


def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
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

            # Proactively convert object columns to datetime
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                    except (ValueError, TypeError):
                        continue
            
            date_cols_for_dropna = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if date_cols_for_dropna:
                df.dropna(subset=date_cols_for_dropna, inplace=True)

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
        if st.button("Начать заново", key="error_reset_1"):
            reset_session()
            st.rerun()
        st.stop()

    time_col = st.selectbox("1. Выберите колонку с датой/временем:", date_cols, key="time_col_selector")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    available_value_cols = [col for col in numeric_cols if col != time_col]

    if not available_value_cols:
        st.error("В файле не найдено числовых колонок для анализа.")
        if st.button("Начать заново", key="error_reset_2"):
            reset_session()
            st.rerun()
        st.stop()

    value_col = st.selectbox("2. Выберите колонку со значениями:", available_value_cols, key="value_col_selector")
    
    st.session_state.time_col = time_col
    st.session_state.value_col = value_col

    available_extra_cols = [col for col in numeric_cols if col not in [time_col, value_col]]
    
    st.subheader("Настройки анализа")
    col1, col2 = st.columns(2)

    with col1:
        extra_cols = st.multiselect("3. Выберите доп. факторы (только числовые):", available_extra_cols, key="extra_cols_selector")
        test_size = st.slider("4. Размер тестовой выборки (%):", 20, 50, 25, 5)
        use_optuna = st.toggle("Использовать Optuna?", value=False, help="Может улучшить точность, но значительно дольше.")

    with col2:
        ranking_metric = st.selectbox(
            "5. Метрика для ранжирования:", ["MAPE", "MAE", "R2"], index=0, key="ranking_metric_selector",
            help="- **MAPE**: Ошибка в %.\n- **MAE**: Ошибка в единицах.\n- **R2**: Качество модели (ближе к 1 - лучше)."
        )
        st.success(f"Модели будут отсортированы по **{ranking_metric}**.")

    st.subheader("Анализ временного ряда")
    try:
        df_for_series = df.sort_values(by=time_col).copy()
        series_for_decomp = TimeSeries.from_dataframe(df_for_series, time_col=time_col, value_cols=value_col, fill_missing_dates=True, freq=None)
        series_for_decomp = series_for_decomp.resample(freq='D').mean()
        st.pyplot(plot_decomposition(series_for_decomp, value_col))
    except Exception as e:
        st.warning(f"Не удалось построить график декомпозиции: {e}")

    if st.button("🚀 Начать битву моделей!", type="primary"):
        st.session_state.screen = "results"
        st.session_state.settings = {
            "test_size": test_size, "extra_cols": extra_cols,
            "use_optuna": use_optuna, "ranking_metric": ranking_metric,
        }
        st.rerun()

# --- SCREEN 3: RESULTS ---
elif st.session_state.screen == "results":
    st.title("Шаг 3: Результаты битвы")

    if st.button("↩️ Начать заново", key="reset_button_results"):
        reset_session()
        st.rerun()

    df = st.session_state.df
    settings = st.session_state.settings
    time_col = st.session_state.time_col
    value_col = st.session_state.value_col

    try:
        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

        # --- THE FIX: Force numeric conversion and drop bad rows ---
        cols_to_process = [value_col] + settings["extra_cols"]
        for col in cols_to_process:
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')
        df_sorted.dropna(subset=cols_to_process, inplace=True)
        # --- END FIX ---

        series = TimeSeries.from_dataframe(df_sorted, time_col, value_col, fill_missing_dates=True, freq=None).astype(np.float32)
        test_size_n = int(len(series) * (settings["test_size"] / 100))
        train, val = series[:-test_size_n], series[-test_size_n:]

        future_covariates = None
        if settings["extra_cols"]:
            future_covariates = TimeSeries.from_dataframe(df_sorted, time_col, settings["extra_cols"], fill_missing_dates=True, freq=None).astype(np.float32)

        models_to_run = {name: mi for name, mi in MODELS.items() if not (mi["requires_extras"] and not settings["extra_cols"])}
        results, forecasts = [], {}
        
        progress_bar = st.progress(0, text="Начинаем битву...")
        for i, (name, model_info) in enumerate(models_to_run.items()):
            progress_bar.progress((i + 1) / len(models_to_run), text=f"Обучается: {name}")
            try:
                _, forecast, metrics = train_model(
                    model_name=name, model_info=model_info, train_series=train, val_series=val,
                    use_optuna=settings["use_optuna"], future_covariates=future_covariates,
                )
                results.append({"Модель": name, **metrics})
                forecasts[name] = forecast
            except Exception as e:
                results.append({"Модель": name, "MAPE": "Ошибка", "MAE": "Ошибка", "R2": "Ошибка"})
        progress_bar.empty()

        if not results:
            st.error("Ни одна модель не смогла быть обучена."); st.stop()

        results_df = pd.DataFrame(results).set_index("Модель")
        results_df = results_df.sort_values(by=settings["ranking_metric"], ascending=settings["ranking_metric"] != "R2", na_position='last')
        
        st.session_state.results, st.session_state.forecasts = results_df, forecasts
        
        st.subheader("🏆 Таблица результатов")
        st.dataframe(results_df.style.format("{:.4f}", subset=["MAPE", "MAE", "R2"]))

        st.subheader("📊 График прогнозов")
        models_to_plot = st.multiselect("Выберите модели для отображения:", list(forecasts.keys()), default=list(forecasts.keys()), key="plot_models_selector")
        selected_forecasts = {name: forecasts[name] for name in models_to_plot if name in forecasts}
        fig = plot_forecast(train, val, selected_forecasts)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📥 Выгрузка результатов")
        best_model_name = results_df.index[0]
        best_forecast = forecasts[best_model_name]
        forecast_df = best_forecast.pd_dataframe(); forecast_df.columns = [f"Прогноз ({best_model_name})"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Скачать CSV", forecast_df.to_csv(index=True).encode("utf-8"), f"forecast_{best_model_name}.csv", "text/csv")
        with col2:
            st.download_button("Скачать XLSX", create_excel_download(forecast_df), f"forecast_{best_model_name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col3:
            st.download_button("Скачать PNG", export_fig_to_png(fig), "forecast_plot.png", "image/png")

    except Exception as e:
        st.error(f"Произошла ошибка: {e}"); st.exception(e)
        if st.button("Начать заново", key="error_reset_3"):
            reset_session()
            st.rerun()
