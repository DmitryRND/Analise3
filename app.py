
import streamlit as st
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mape, r2_score, rmse
import warnings
import plotly.graph_objects as go
from models_lib import MODELS, train_model, optimize_hyperparameters
from utils import (
    plot_decomposition,
    plot_forecast,
    plot_final_forecast,
    create_excel_download,
    export_fig_to_png,
    recommend_metric,
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
        "use_hyperopt": False,
        "n_trials": 10,
        "battle_results": None,
        "trained_models": None,
        "forecasts": None,
        "final_forecast": None,
        "manual_date_col": None,
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
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Попытка автоматически найти столбцы с датами
            # Используем более гибкий парсинг дат
            for col in df.columns:
                if df[col].dtype == "object" or 'date' in col.lower() or 'time' in col.lower():
                    try:
                        # Пробуем разные форматы дат
                        parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                        # Если успешно распознано более 80% значений, используем этот столбец
                        if parsed.notna().sum() > len(df) * 0.8:
                            df[col] = parsed
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
    st.dataframe(df.head(5))
    
    # Вывод информации о файле
    st.info(f"📊 **Информация о файле:** Количество строк: {len(df)}, Количество столбцов: {len(df.columns)}")

    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Если даты не найдены автоматически, даем возможность указать вручную
    if not date_cols:
        st.warning("⚠️ Автоматически не найдено колонок с датами.")
        st.subheader("Укажите колонку с датой вручную:")
        manual_date_col = st.selectbox("Выберите столбец с датой:", df.columns.tolist(), 
                                       index=0, key="manual_date_select")
        
        if st.button("Проверить и использовать эту колонку"):
            try:
                # Пробуем преобразовать в дату с разными форматами
                test_col = pd.to_datetime(df[manual_date_col], infer_datetime_format=True, errors='coerce')
                if test_col.notna().sum() > len(df) * 0.8:  # Если больше 80% успешно преобразовано
                    df[manual_date_col] = test_col
                    date_cols = [manual_date_col]
                    st.session_state.df = df
                    st.success(f"✅ Колонка '{manual_date_col}' успешно распознана как дата!")
                    st.rerun()
                else:
                    st.error(f"❌ Не удалось распознать даты в колонке '{manual_date_col}'. Убедитесь, что формат даты корректен.")
            except Exception as e:
                st.error(f"❌ Ошибка при преобразовании даты: {e}")
        st.stop()
    
    # Выбор колонки с датой
    default_idx = 0
    if st.session_state.time_col in date_cols:
        default_idx = date_cols.index(st.session_state.time_col)
    st.session_state.time_col = st.selectbox("1. Выберите колонку с датой/временем:", date_cols, index=default_idx)

    # Получаем числовые колонки и колонки, которые можно преобразовать в числа
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Проверяем остальные колонки на возможность преобразования в числа
    potentially_numeric = []
    for col in df.columns:
        if col not in date_cols and col not in numeric_cols:
            # Пробуем преобразовать в число (учитывая запятые, пробелы и т.д.)
            test_series = df[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False).str.replace('$', '', regex=False)
            try:
                pd.to_numeric(test_series, errors='raise')
                potentially_numeric.append(col)
            except (ValueError, TypeError):
                pass
    
    # Объединяем все доступные числовые колонки
    all_numeric_cols = numeric_cols + potentially_numeric
    available_value_cols = [col for col in all_numeric_cols if col != st.session_state.time_col]

    if not available_value_cols:
        st.error("В файле не найдено числовых колонок для анализа.")
        if st.button("Начать заново"): reset_session(); st.rerun()
        st.stop()

    st.session_state.value_col = st.selectbox("2. Выберите колонку со значениями:", available_value_cols, index=available_value_cols.index(st.session_state.value_col) if st.session_state.value_col in available_value_cols else 0)
    
    available_extra_cols = [col for col in all_numeric_cols if col not in [st.session_state.time_col, st.session_state.value_col]]
    st.session_state.extra_cols = st.multiselect("3. Выберите доп. факторы (необязательно):", available_extra_cols, default=st.session_state.extra_cols)

    st.subheader("Настройки анализа")
    
    df_for_series = df.sort_values(by=st.session_state.time_col).copy()
    # Очищаем числовые данные от запятых и других разделителей
    if df_for_series[st.session_state.value_col].dtype == 'object':
        df_for_series[st.session_state.value_col] = df_for_series[st.session_state.value_col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    df_for_series[st.session_state.value_col] = pd.to_numeric(df_for_series[st.session_state.value_col], errors='coerce')
    df_for_series.dropna(subset=[st.session_state.value_col], inplace=True)
    
    # Определяем частоту временного ряда перед созданием TimeSeries
    if len(df_for_series) < 3:
        st.error(f"Недостаточно данных для анализа. Требуется минимум 3 строки, а у вас {len(df_for_series)}.")
        st.stop()
    
    # Пробуем определить частоту автоматически
    df_indexed = df_for_series.set_index(st.session_state.time_col).sort_index()
    inferred_freq = pd.infer_freq(df_indexed.index)
    
    # Если частота не определена, пробуем определить по разнице между датами
    if inferred_freq is None and len(df_indexed) > 1:
        time_diffs = df_indexed.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            median_diff = time_diffs.median()
            # Определяем частоту по медианной разнице
            if pd.Timedelta(hours=23) <= median_diff <= pd.Timedelta(hours=25):
                inferred_freq = 'D'  # Дневная
            elif pd.Timedelta(days=27) <= median_diff <= pd.Timedelta(days=32):
                inferred_freq = 'M'  # Месячная
            elif pd.Timedelta(hours=11) <= median_diff <= pd.Timedelta(hours=13):
                inferred_freq = 'H'  # Часовая
            else:
                inferred_freq = None
    
    # Создаем TimeSeries с определенной частотой или без заполнения пропусков
    if inferred_freq:
        series = TimeSeries.from_dataframe(df_for_series, time_col=st.session_state.time_col, value_cols=st.session_state.value_col, fill_missing_dates=True, freq=inferred_freq)
    else:
        # Если частоту определить не удалось, создаем без заполнения пропусков
        series = TimeSeries.from_dataframe(df_for_series, time_col=st.session_state.time_col, value_cols=st.session_state.value_col, fill_missing_dates=False)
    
    # Показываем информацию о частоте
    series_freq = pd.infer_freq(series.time_index)
    if series_freq or inferred_freq:
        freq_display = series_freq if series_freq else inferred_freq
        st.info(f"Определенная частота временного ряда: {freq_display}")
        # Автоматически устанавливаем период сезонности для дневных данных
        if inferred_freq == 'D' and st.session_state.get('freq_set_auto', False) is False:
            st.session_state.season_period = 7
            st.session_state.freq_set_auto = True
            st.rerun()
        elif inferred_freq == 'H' and st.session_state.get('freq_set_auto', False) is False:
            st.session_state.season_period = 24
            st.session_state.freq_set_auto = True
            st.rerun()
    else:
        st.warning("Не удалось автоматически определить частоту ряда. Период сезонности нужно установить вручную.")

    st.session_state.n_forecast = st.number_input("4. Укажите срок прогнозирования (в шагах):", min_value=1, value=st.session_state.n_forecast, step=1)
    st.session_state.season_period = st.number_input("5. Укажите период сезонности:", min_value=2, value=st.session_state.season_period, step=1)
    
    # Рекомендация метрики
    st.subheader("Метрика для ранжирования")
    metric_rec = recommend_metric(series)
    st.session_state.ranking_metric = metric_rec["metric"]
    st.info(f"Рекомендуемая метрика: **{metric_rec['metric']}**")
    st.markdown(f"**Пояснение:** {metric_rec['reason']}")
    
    # График декомпозиции
    st.subheader("📈 Анализ сезонности, тренда и выпадов")
    if len(series) < 2 * st.session_state.season_period:
        st.warning(f"Недостаточно данных для анализа сезонности. Требуется как минимум {2 * st.session_state.season_period} точек данных (2 периода), а у вас {len(series)}. График не будет построен.")
    else:
        try:
            fig_decomp = plot_decomposition(series, period=st.session_state.season_period)
            st.plotly_chart(fig_decomp, width='stretch')
        except Exception as e:
            st.warning(f"Не удалось построить график декомпозиции: {e}")
    
    # Настройки подбора гиперпараметров
    st.subheader("⚙️ Настройки обучения моделей")
    st.session_state.use_hyperopt = st.checkbox("Использовать подбор гиперпараметров", value=st.session_state.use_hyperopt)
    if st.session_state.use_hyperopt:
        st.session_state.n_trials = st.slider("Количество подходов для подбора гиперпараметров:", min_value=5, max_value=50, value=st.session_state.n_trials, step=5)
        st.caption("⚠️ Внимание: подбор гиперпараметров значительно увеличивает время обучения моделей.")

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
            
            # Очищаем и преобразуем числовые колонки
            for col in cols_to_process:
                # Убираем запятые, пробелы и другие разделители тысяч
                if df_sorted[col].dtype == 'object':
                    df_sorted[col] = df_sorted[col].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False).str.replace('$', '', regex=False)
                df_sorted[col] = pd.to_numeric(df_sorted[col], errors='coerce')
            
            # Удаляем строки с NaN в основных колонках
            df_sorted.dropna(subset=[value_col], inplace=True)
            
            # Определяем частоту для этого датасета тоже
            df_indexed = df_sorted.set_index(time_col).sort_index()
            inferred_freq = pd.infer_freq(df_indexed.index)
            
            if inferred_freq is None and len(df_indexed) > 1:
                time_diffs = df_indexed.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    if pd.Timedelta(hours=23) <= median_diff <= pd.Timedelta(hours=25):
                        inferred_freq = 'D'
                    elif pd.Timedelta(days=27) <= median_diff <= pd.Timedelta(days=32):
                        inferred_freq = 'M'
                    elif pd.Timedelta(hours=11) <= median_diff <= pd.Timedelta(hours=13):
                        inferred_freq = 'H'
            
            # Создаем TimeSeries с правильной частотой
            if inferred_freq and len(df_sorted) >= 3:
                series = TimeSeries.from_dataframe(df_sorted, time_col, value_col, fill_missing_dates=True, freq=inferred_freq).astype(np.float32)
            elif len(df_sorted) >= 3:
                series = TimeSeries.from_dataframe(df_sorted, time_col, value_col, fill_missing_dates=False).astype(np.float32)
            else:
                st.error(f"Недостаточно данных для анализа. Требуется минимум 3 строки, а у вас {len(df_sorted)}.")
                st.stop()
            
            # FIX: Add a strict check for minimum training size to prevent IndexError
            min_train_size = 10 
            if (len(series) - n_forecast) < min_train_size:
                st.error(f"Ошибка: Недостаточно данных для обучения. Требуется как минимум {min_train_size} точек данных для обучения, но после выделения горизонта прогноза ({n_forecast}) остается только {len(series) - n_forecast}. Пожалуйста, уменьшите срок прогноза или загрузите больше данных.")
                st.stop()

            train, val = series[:-n_forecast], series[-n_forecast:]
            
            future_covariates = None
            if extra_cols:
                # Очищаем экзогенные переменные от NaN
                for col in extra_cols:
                    if df_sorted[col].isna().any():
                        df_sorted[col].ffill(inplace=True)
                        df_sorted[col].bfill(inplace=True)
                    # Если все еще есть NaN, заполняем нулями
                    df_sorted[col].fillna(0, inplace=True)
                
                # Создаем экзогенные переменные с той же частотой
                try:
                    # Убеждаемся, что экзогенные переменные имеют ту же частоту и временной индекс
                    if inferred_freq:
                        # Используем тот же временной диапазон и частоту, что и основной ряд
                        # Важно: используем fill_missing_dates=True с freq для корректной работы
                        # Сначала создаем без заполнения, чтобы проверить частоту
                        try:
                            future_covariates = TimeSeries.from_dataframe(
                                df_sorted, 
                                time_col, 
                                extra_cols, 
                                fill_missing_dates=True, 
                                freq=inferred_freq
                            ).astype(np.float32)
                        except ValueError as e:
                            # Если freq не работает, пробуем без явного freq
                            future_covariates = TimeSeries.from_dataframe(
                                df_sorted, 
                                time_col, 
                                extra_cols, 
                                fill_missing_dates=True,
                                freq=None
                            ).astype(np.float32)
                            # Устанавливаем freq вручную если возможно
                            try:
                                future_covariates = future_covariates.with_freq(inferred_freq) if inferred_freq else future_covariates
                            except:
                                pass
                        
                        # Дополняем экзогенные переменные для периода прогноза (используем последние значения)
                        if len(future_covariates) < len(series) + n_forecast:
                            from utils import _get_ts_values_and_index
                            last_values, last_index = _get_ts_values_and_index(future_covariates)
                            last_vals = last_values[-1] if len(last_values.shape) == 1 else last_values[-1, :]
                            
                            # Создаем дополнительные даты на основе частоты
                            last_date = last_index[-1]
                            if inferred_freq == 'D':
                                freq_timedelta = pd.Timedelta(days=1)
                            elif inferred_freq == 'M' or inferred_freq.startswith('M'):
                                freq_timedelta = pd.Timedelta(days=30)
                            elif inferred_freq == 'H':
                                freq_timedelta = pd.Timedelta(hours=1)
                            else:
                                freq_timedelta = pd.Timedelta(days=1)
                            
                            needed_dates = len(series) + n_forecast - len(future_covariates)
                            extended_dates = pd.date_range(start=last_date + freq_timedelta, periods=needed_dates, freq=inferred_freq)
                            
                            # Формируем значения (для нескольких экзогенных переменных)
                            if len(last_vals.shape) == 0:
                                extended_values = np.tile(last_vals, (len(extended_dates),))
                            else:
                                extended_values = np.tile(last_vals, (len(extended_dates), 1))
                            
                            extended_ts = TimeSeries.from_times_and_values(extended_dates, extended_values)
                            future_covariates = future_covariates.concatenate(extended_ts)
                    else:
                        # Без частоты просто создаем без заполнения пропусков
                        future_covariates = TimeSeries.from_dataframe(df_sorted, time_col, extra_cols, fill_missing_dates=False).astype(np.float32)
                except Exception as e:
                    st.warning(f"Не удалось создать экзогенные переменные: {e}. Модели будут работать без них.")
                    future_covariates = None

            # Выбор моделей согласно требованиям
            # Базовые модели (работают без экзогенных): ExponentialSmoothing, LinearRegression, Prophet, AutoARIMA, LightGBM, Theta, CatBoost
            # Сложные модели (только с экзогенными): FFT, N-BEATS и другие
            base_models = ["ExponentialSmoothing", "LinearRegression", "Prophet", "AutoARIMA", "LightGBM", "Theta"]
            # Добавляем CatBoost, если он доступен
            if "CatBoost" in MODELS:
                base_models.append("CatBoost")
            advanced_models = ["FFT", "N-BEATS"]
            
            if extra_cols:
                # Если есть экзогенные переменные - используем все модели
                models_to_run = {name: mi for name, mi in MODELS.items()}
            else:
                # Если нет экзогенных - только базовые
                models_to_run = {name: mi for name, mi in MODELS.items() if name in base_models}
            
            results_list, forecasts, trained_models, best_params_dict = [], {}, {}, {}
            total_steps = len(models_to_run) * (2 if st.session_state.use_hyperopt else 1)
            current_step = 0
            
            # Выводим индикатор прогресса на передний план
            status_info = st.info("🔄 **Начинаем обучение моделей...**")
            progress_bar = st.progress(0, text="⏳ Подготовка к обучению моделей...")
            status_text = st.empty()
            
            for name, model_info in models_to_run.items():
                # Проверка совместимости модели с экзогенными переменными
                if model_info["requires_extras"] and not extra_cols:
                    # Пропускаем модели, которые требуют экзогенные переменные
                    continue
                
                # Подбор гиперпараметров если включен
                best_params = None
                if st.session_state.use_hyperopt:
                    current_step += 1
                    progress = current_step / total_steps
                    progress_bar.progress(progress, text=f"🔍 Подбор гиперпараметров: {name}")
                    status_text.info(f"**Текущий этап:** Подбор гиперпараметров для модели {name} ({current_step}/{total_steps})")
                    best_params, opt_error = optimize_hyperparameters(
                        model_name=name,
                        train_series=train,
                        val_series=val,
                        forecast_horizon=len(val),
                        future_covariates=future_covariates,
                        n_trials=st.session_state.n_trials,
                        metric=st.session_state.ranking_metric.lower()
                    )
                    if opt_error:
                        st.warning(f"Не удалось оптимизировать {name}: {opt_error}. Используются параметры по умолчанию.")
                    else:
                        best_params_dict[name] = best_params
                
                # Обучение модели
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress, text=f"🚀 Обучается: {name}")
                status_text.info(f"**Текущий этап:** Обучение модели {name} ({current_step}/{total_steps})")
                
                forecast, model, error = train_model(
                    model_name=name, 
                    train_series=train,
                    forecast_horizon=len(val), 
                    future_covariates=future_covariates,
                    model_params=best_params if best_params else None
                )

                if error or forecast is None:
                    results_list.append({"Модель": name, "MAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Гиперпараметры": error or "Неизвестная ошибка"})
                    continue

                # Получаем значения для проверки нулей
                from utils import _get_ts_values_and_index
                val_values, _ = _get_ts_values_and_index(val)
                mape_score = mape(val, forecast) if 0 not in val_values else np.nan
                mae_score = mae(val, forecast)
                r2_score_val = r2_score(val, forecast)
                rmse_score = rmse(val, forecast)
                
                # Сохраняем информацию о гиперпараметрах
                if best_params:
                    params_str = f"Оптимизированные: {best_params}"
                else:
                    try:
                        params = model.model_params if hasattr(model, 'model_params') else getattr(model, 'model', {}).get_params() if hasattr(model, 'model') else {}
                        params_str = str(params) if params else "По умолчанию"
                    except:
                        params_str = "По умолчанию"
                
                results_list.append({"Модель": name, "MAPE": mape_score, "MAE": mae_score, "RMSE": rmse_score, "R2": r2_score_val, "Гиперпараметры": params_str})
                forecasts[name] = forecast
                trained_models[name] = model
            
            progress_bar.empty()
            status_text.empty()
            status_info.empty()  # Убираем информационное сообщение о начале обучения
            st.success("✅ Обучение всех моделей завершено!")

            if not results_list:
                st.error("Ни одна модель не смогла быть обучена."); st.stop()

            results_df = pd.DataFrame(results_list).set_index("Модель")
            # Определяем, в каком порядке сортировать:
            # для R2 и RMSE по убыванию (больше лучше), для остальных по возрастанию.
            ascending_flag = False if st.session_state.ranking_metric == "R2" else True


            results_df = results_df.sort_values(
                by=st.session_state.ranking_metric,
                ascending=ascending_flag,
                na_position='last'
            )
            
            # Создаем финальный прогноз лучшей модели на весь период
            best_model_name = results_df.dropna(subset=[st.session_state.ranking_metric]).index[0]
            best_model = trained_models.get(best_model_name)
            final_forecast = None
            
            if best_model is not None:
                try:
                    # Используем оптимальные параметры если они были
                    model_params = best_params_dict.get(best_model_name, {})
                    
                    # Обучаем на всех данных и прогнозируем на n_forecast шагов вперед
                    forecast_result, _, error = train_model(
                        model_name=best_model_name,
                        train_series=series,
                        forecast_horizon=n_forecast,
                        future_covariates=future_covariates,
                        model_params=model_params if model_params else None
                    )
                    if not error and forecast_result is not None:
                        final_forecast = forecast_result
                except Exception as e:
                    st.warning(f"Не удалось создать финальный прогноз: {e}")
            
            st.session_state.battle_results = results_df
            st.session_state.forecasts = forecasts
            st.session_state.trained_models = trained_models
            st.session_state.final_forecast = final_forecast

        except Exception as e:
            st.error(f"Произошла критическая ошибка на этапе выполнения: {e}")
            st.exception(e)
            st.stop()

    results_df = st.session_state.battle_results
    forecasts = st.session_state.forecasts
    
    st.subheader(f"🏆 Таблица результатов (прогноз на {st.session_state.n_forecast} шагов)")
    st.markdown(f"Ранжирование по: **{st.session_state.ranking_metric}**")

    def highlight_best(s):
        is_min = s.name in ["MAE", "MAPE", "RMSE"]
        best_val = s.min() if is_min else s.max()
        return ['background-color: #28a745' if v == best_val else '' for v in s]

    st.dataframe(results_df.style.apply(highlight_best, subset=["MAE", "MAPE", "RMSE", "R2"]).format({"MAPE": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}, na_rep="-"))

    # График прогнозов на тестовых данных
    st.subheader("📊 График прогнозов моделей на тестовых данных")
    successful_models = list(forecasts.keys())
    if successful_models:
        default_models = results_df.dropna(subset=[st.session_state.ranking_metric]).head(3).index.tolist()
        
        models_to_plot = st.multiselect("Выберите модели для отображения:", successful_models, default=default_models)
        
        if models_to_plot:
            df_for_plot = st.session_state.df.sort_values(by=st.session_state.time_col).copy()
            df_for_plot[st.session_state.value_col] = pd.to_numeric(df_for_plot[st.session_state.value_col], errors='coerce')
            df_for_plot.dropna(subset=[st.session_state.value_col], inplace=True)
            
            # Определяем частоту для графика тоже
            df_plot_indexed = df_for_plot.set_index(st.session_state.time_col).sort_index()
            plot_freq = pd.infer_freq(df_plot_indexed.index)
            if plot_freq is None and len(df_plot_indexed) > 1:
                time_diffs = df_plot_indexed.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    if pd.Timedelta(hours=23) <= median_diff <= pd.Timedelta(hours=25):
                        plot_freq = 'D'
                    elif pd.Timedelta(days=27) <= median_diff <= pd.Timedelta(days=32):
                        plot_freq = 'M'
                    elif pd.Timedelta(hours=11) <= median_diff <= pd.Timedelta(hours=13):
                        plot_freq = 'H'
            
            if plot_freq:
                series_to_plot = TimeSeries.from_dataframe(df_for_plot, st.session_state.time_col, st.session_state.value_col, fill_missing_dates=True, freq=plot_freq).astype(np.float32)
            else:
                series_to_plot = TimeSeries.from_dataframe(df_for_plot, st.session_state.time_col, st.session_state.value_col, fill_missing_dates=False).astype(np.float32)
            train_plot, val_plot = series_to_plot[:-st.session_state.n_forecast], series_to_plot[-st.session_state.n_forecast:]
            
            selected_forecasts = {name: forecasts[name] for name in models_to_plot if name in forecasts}
            fig_test = plot_forecast(train_plot, val_plot, selected_forecasts)
            st.plotly_chart(fig_test, width='stretch')
    else:
        st.warning("Ни одна модель не смогла построить прогноз, график недоступен.")
    
    # График финального прогноза лучшей модели
    st.subheader("🎯 Финальный прогноз на нужный период (лучшая модель)")
    best_model_name = results_df.dropna(subset=[st.session_state.ranking_metric]).index[0]
    if st.session_state.final_forecast is not None:
        df_for_final = st.session_state.df.sort_values(by=st.session_state.time_col).copy()
        df_for_final[st.session_state.value_col] = pd.to_numeric(df_for_final[st.session_state.value_col], errors='coerce')
        df_for_final.dropna(subset=[st.session_state.value_col], inplace=True)
        # Используем ту же логику определения частоты
        df_final_indexed = df_for_final.set_index(st.session_state.time_col).sort_index()
        final_freq = pd.infer_freq(df_final_indexed.index)
        if final_freq is None and len(df_final_indexed) > 1:
            time_diffs = df_final_indexed.index.to_series().diff().dropna()
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                if pd.Timedelta(hours=23) <= median_diff <= pd.Timedelta(hours=25):
                    final_freq = 'D'
                elif pd.Timedelta(days=27) <= median_diff <= pd.Timedelta(days=32):
                    final_freq = 'M'
                elif pd.Timedelta(hours=11) <= median_diff <= pd.Timedelta(hours=13):
                    final_freq = 'H'
        
        if final_freq:
            series_full = TimeSeries.from_dataframe(df_for_final, st.session_state.time_col, st.session_state.value_col, fill_missing_dates=True, freq=final_freq).astype(np.float32)
        else:
            series_full = TimeSeries.from_dataframe(df_for_final, st.session_state.time_col, st.session_state.value_col, fill_missing_dates=False).astype(np.float32)
        
        fig_final = plot_final_forecast(series_full, st.session_state.final_forecast)
        st.plotly_chart(fig_final, width='stretch')
        
        # Выгрузка результатов
        st.subheader("📥 Выгрузка результатов")
        col1, col2, col3 = st.columns(3)
        with col1:
            # Конвертируем TimeSeries в DataFrame для CSV
            from utils import _get_ts_dataframe
            forecast_df = _get_ts_dataframe(st.session_state.final_forecast)
            st.download_button(
                "Скачать CSV (прогноз)", 
                forecast_df.to_csv(index=True).encode("utf-8"), 
                f"forecast_{best_model_name}.csv", 
                "text/csv"
            )
        with col2:
            st.download_button(
                "Скачать XLSX (прогноз)", 
                create_excel_download(st.session_state.final_forecast), 
                f"forecast_{best_model_name}.xlsx", 
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col3:
            st.download_button(
                "Скачать PNG (график)", 
                export_fig_to_png(fig_final), 
                "forecast_plot.png", 
                "image/png"
            )
    else:
        st.warning("Не удалось создать финальный прогноз для выгрузки.")
