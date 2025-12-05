
import streamlit as st
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.metrics import mae, mape, r2_score, rmse, mse
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import streamlit.components.v1 as components
import time
import os

try:
    import psutil
except ImportError:
    psutil = None
pio.templates.default = "plotly_dark"
from models_lib import MODELS, train_model, optimize_hyperparameters
from utils import (
    plot_decomposition,
    plot_forecast,
    plot_final_forecast,
    create_excel_download,
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

# --- Helpers ---
MAX_ROWS = 5000

# Ограничиваем число потоков для матричных библиотек на слабых серверах
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def adjust_daily_to_monthly(freq, index):
    """
    Возвращаем исходную частоту без принудительных преобразований.
    """
    return freq, False  # None или найденная частота без подмены

def normalize_month_start(df, time_col, freq):
    """
    Без преобразований — вернём как есть.
    """
    return df

def safe_timeseries_from_df(df, time_col, value_col, freq, label=""):
    """
    Создаёт TimeSeries с попыткой заполнить пропуски по freq, при ошибках пробует freq=None,
    и в финале строит без fill_missing_dates.
    """
    try:
        return TimeSeries.from_dataframe(
            df,
            time_col=time_col,
            value_cols=value_col,
            fill_missing_dates=True,
            freq=freq,
        )
    except Exception as e1:
        if label:
            st.warning(f"Не удалось установить частоту ({freq}) для {label}: {e1}. Пробую без freq.")
        try:
            return TimeSeries.from_dataframe(
                df,
                time_col=time_col,
                value_cols=value_col,
                fill_missing_dates=True,
                freq=None,
            )
        except Exception as e2:
            if label:
                st.warning(f"Не удалось заполнить даты для {label} даже без freq: {e2}. Строю без fill_missing_dates.")
            return TimeSeries.from_dataframe(
                df,
                time_col=time_col,
                value_cols=value_col,
                fill_missing_dates=False,
            )

def render_resource_panel(start_time=None):
    """Показываем быструю информацию о ресурсах в сайдбаре."""
    with st.sidebar:
        st.markdown("### Мониторинг")
        if psutil:
            proc = psutil.Process()
            cpu = psutil.cpu_percent(interval=0.1)
            mem = proc.memory_info().rss / (1024 ** 2)
            sys_mem = psutil.virtual_memory()
            st.write(f"CPU: {cpu:.1f}%")
            st.write(f"RAM: {mem:.1f} MB / {(sys_mem.total/(1024**3)):.1f} GB")
        else:
            st.write("psutil не установлен")
        if start_time:
            elapsed = time.time() - start_time
            st.write(f"Время с запуска: {elapsed/60:.1f} мин")

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
        "ranking_metric_user_set": False,
        "use_hyperopt": False,
        "n_trials": 10,
        "battle_results": None,
        "trained_models": None,
        "forecasts": None,
        "final_forecast": None,
        "manual_date_col": None,
        "scroll_to_top": False,
        "val_size": None,
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

# --- Scroll helper ---
if st.session_state.get("scroll_to_top"):
    components.html(
        """
        <script>
            (() => {
                const goTop = () => {
                    try { window.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                    try { parent.window.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                    try { document.documentElement.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                    try { document.body.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                };
                requestAnimationFrame(goTop);
                setTimeout(goTop, 50);
                setTimeout(goTop, 150);
            })();
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state.scroll_to_top = False

# --- SCREEN 1: UPLOAD ---
if st.session_state.screen == "upload":
    st.title("⚔️ Битва моделей временных рядов")
    st.header("Шаг 1: Загрузите ваш файл")
    st.info("Перед загрузкой убедитесь, что временной ряд предобработан: даты приведены к нужной частоте, пропуски заполнены, дубликаты удалены.")

    uploaded_file = st.file_uploader(
        "Выберите CSV или Excel файл", type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Ограничение по строкам для слабых серверов
            if len(df) > MAX_ROWS:
                st.warning(f"Файл содержит {len(df)} строк, лимит — {MAX_ROWS}. На слабом сервере это может упасть.")
                if not st.button("Продолжить несмотря на лимит", key="continue_over_limit"):
                    st.stop()

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
                # Пробуем интерпретировать колонку года (формат YYYY)
                if str(col).lower() in ["year", "год"] or pd.api.types.is_integer_dtype(df[col]):
                    parsed_year = pd.to_datetime(df[col], format="%Y", errors="coerce")
                    if parsed_year.notna().sum() > len(df) * 0.8:
                        df[col] = parsed_year

            # Проверяем пропуски и предлагаем заполнить на этапе загрузки
            total_missing = int(df.isna().sum().sum())
            if total_missing > 0:
                miss_cols = df.isna().sum()
                miss_cols = miss_cols[miss_cols > 0].to_dict()
                st.warning(f"В файле обнаружены пропуски ({total_missing} значений). Колонки с пропусками: {miss_cols}")
                if st.button("Заполнить пропуски автоматически", key="fill_missing_upload"):
                    df_filled = df.copy()
                    num_cols = df_filled.select_dtypes(include=[np.number]).columns
                    for col in num_cols:
                        if df_filled[col].isna().any():
                            df_filled[col] = df_filled[col].interpolate(limit_direction="both")
                            df_filled[col] = df_filled[col].fillna(df_filled[col].mean(skipna=True))
                    # Остальные колонки заполняем предыдущими/следующими значениями
                    other_cols = [c for c in df_filled.columns if c not in num_cols]
                    if other_cols:
                        df_filled[other_cols] = df_filled[other_cols].ffill().bfill()
                    df = df_filled
                    st.success("Пропуски заполнены автоматически.")
            
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

    if len(df) > 500:
        st.warning("⚠️ Файл содержит более 500 строк. Обучение и расчёты могут занять заметно больше времени.")

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
    # Если выбрана колонка года — переводим в даты (начало года)
    if not pd.api.types.is_datetime64_any_dtype(df_for_series[st.session_state.time_col]):
        if str(st.session_state.time_col).lower() in ["year", "год"]:
            df_for_series[st.session_state.time_col] = pd.to_datetime(df_for_series[st.session_state.time_col], format="%Y", errors="coerce")
    
    # Удаляем дубликаты по времени
    dup_count = df_for_series.duplicated(subset=[st.session_state.time_col]).sum()
    if dup_count > 0:
        st.warning(f"Обнаружены дубликаты по времени ({dup_count}). Они будут удалены.")
        df_for_series = df_for_series.drop_duplicates(subset=[st.session_state.time_col], keep="first")

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
    inferred_freq, monthly_forced = adjust_daily_to_monthly(inferred_freq, df_indexed.index)
    
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
    
            if not monthly_forced:
                inferred_freq, monthly_forced = adjust_daily_to_monthly(inferred_freq, df_indexed.index)
    
    # Создаем TimeSeries с определенной частотой или без заполнения пропусков
    df_for_series = normalize_month_start(df_for_series, st.session_state.time_col, inferred_freq)
    series = safe_timeseries_from_df(
        df_for_series,
        time_col=st.session_state.time_col,
        value_col=st.session_state.value_col,
        freq=inferred_freq,
        label="основного ряда",
    )
    
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

    # Проверка и заполнение пропусков в выбранных колонках
    cols_to_check = [st.session_state.value_col] + st.session_state.extra_cols
    missing_counts = {col: df_for_series[col].isna().sum() for col in cols_to_check}
    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        st.warning(f"Обнаружены пропуски в данных: {missing_counts}")

        def has_trend(series_vals):
            vals = series_vals.dropna().to_numpy()
            if len(vals) < 3:
                return False
            x = np.arange(len(vals))
            slope = np.polyfit(x, vals, 1)[0]
            std = np.std(vals) + 1e-8
            return abs(slope) / std > 0.05

        trend_present = has_trend(df_for_series[st.session_state.value_col])
        suggested_method = "интерполяцией" if trend_present else "средним"
        st.info(f"Предлагаем заполнить пропуски {suggested_method} (тренд {'обнаружен' if trend_present else 'не обнаружен'}).")

        if st.button("Заполнить пропуски автоматически"):
            for col in cols_to_check:
                if df_for_series[col].isna().any():
                    if trend_present:
                        df_for_series[col] = df_for_series[col].interpolate(limit_direction="both")
                    else:
                        mean_val = df_for_series[col].mean(skipna=True)
                        df_for_series[col] = df_for_series[col].fillna(mean_val)
            st.success("Пропуски заполнены.")
    # Страховка: если остались NaN, заполняем последним значением, затем средним
    if df_for_series[cols_to_check].isna().any().any():
        df_for_series[cols_to_check] = df_for_series[cols_to_check].ffill().bfill()
        for col in cols_to_check:
            if df_for_series[col].isna().any():
                df_for_series[col] = df_for_series[col].fillna(df_for_series[col].mean(skipna=True))

    # Проверка пропусков по временной шкале и кнопка для заполнения дат
    freq_guess = pd.infer_freq(df_for_series.sort_values(by=st.session_state.time_col)[st.session_state.time_col])
    if not freq_guess:
        diffs = df_for_series[st.session_state.time_col].sort_values().diff().dropna()
        if not diffs.empty:
            median_diff = diffs.median()
            if pd.Timedelta(days=27) <= median_diff <= pd.Timedelta(days=32):
                freq_guess = "M"
            elif pd.Timedelta(days=360) <= median_diff <= pd.Timedelta(days=380):
                freq_guess = "A"
    if freq_guess:
        full_index = pd.date_range(
            start=df_for_series[st.session_state.time_col].min(),
            end=df_for_series[st.session_state.time_col].max(),
            freq=freq_guess,
        )
        missing_dates = full_index.difference(df_for_series[st.session_state.time_col])
        if len(missing_dates) > 0:
            st.warning(f"Обнаружены пропущенные даты: {len(missing_dates)} точек. Частота: {freq_guess}")
            if st.button("Заполнить пропущенные даты", key="fill_missing_dates"):
                df_tmp = df_for_series.set_index(st.session_state.time_col).reindex(full_index)
                # интерполяция числовых колонок
                for col in cols_to_check:
                    df_tmp[col] = df_tmp[col].interpolate(limit_direction="both")
                    df_tmp[col] = df_tmp[col].fillna(df_tmp[col].mean(skipna=True))
                df_tmp = df_tmp.ffill().bfill()
                df_for_series = df_tmp.reset_index().rename(columns={"index": st.session_state.time_col})
                st.success("Пропущенные даты заполнены.")
    
    # Рекомендация метрики + выбор (по умолчанию рекомендуемая)
    st.subheader("Метрика для ранжирования")
    metric_rec = recommend_metric(series)
    metric_help = {
        "MAE": "Средняя абсолютная ошибка, устойчива к выбросам и нулям.",
        "MAPE": "Средняя процентная ошибка, только для положительных значений, удобна в процентах.",
        "RMSE": "Корень из среднеквадратичной ошибки, сильнее наказывает крупные промахи.",
        "MSE": "Среднеквадратичная ошибка, квадрат единиц, жёстко штрафует большие ошибки.",
    }
    metric_options = list(metric_help.keys())
    recommended_metric = metric_rec["metric"] if metric_rec["metric"] in metric_options else "MAE"

    # Автоматически ставим рекомендованную метрику, пока пользователь не выбрал вручную
    if not st.session_state.get("ranking_metric_user_set"):
        st.session_state.ranking_metric = recommended_metric

    current_metric = (
        st.session_state.ranking_metric
        if st.session_state.ranking_metric in metric_options
        else recommended_metric
    )

    chosen_metric = st.selectbox(
        "Выберите метрику ранжирования моделей:",
        metric_options,
        index=metric_options.index(current_metric),
        key="ranking_metric_select",
        help="Используется для сортировки результатов. По умолчанию — рекомендованная системой.",
    )
    if chosen_metric != st.session_state.ranking_metric:
        st.session_state.ranking_metric_user_set = True
    st.session_state.ranking_metric = chosen_metric
    st.info(f"Рекомендуемая метрика: **{recommended_metric}**")
    st.markdown(f"**Пояснение:** {metric_rec['reason']}")
    st.caption("\n".join([f"- **{k}**: {v}" for k, v in metric_help.items()]))
    
    # График декомпозиции
    st.subheader("📈 Анализ сезонности, тренда и выпадов")
    if len(series) < 2 * st.session_state.season_period:
        st.warning(f"Недостаточно данных для анализа сезонности. Требуется как минимум {2 * st.session_state.season_period} точек данных (2 периода), а у вас {len(series)}. График не будет построен.")
    else:
        # Если годовая частота — сезонность обычно отсутствует; пропускаем график
        freq_str = str(getattr(series, "freq", "")) if hasattr(series, "freq") else ""
        if freq_str.upper().startswith(("A", "Y")):
            st.info("Годовая частота обнаружена: график сезонности пропущен.")
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
        st.session_state.scroll_to_top = True
        st.session_state.screen = "results"
        st.rerun()

# --- SCREEN 3: RESULTS ---
elif st.session_state.screen == "results":
    # Мониторинг ресурсов
    render_resource_panel(st.session_state.get("run_start_time"))
    # Always ensure we scroll to top when entering results
    components.html(
        """
        <script>
            (() => {
                const goTop = () => {
                    try { window.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                    try { parent.window.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                    try { document.documentElement.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                    try { document.body.scrollTo({top: 0, behavior: 'smooth'}); } catch(e) {}
                };
                requestAnimationFrame(goTop);
                setTimeout(goTop, 50);
                setTimeout(goTop, 150);
            })();
        </script>
        """,
        height=0,
        width=0,
    )
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
        st.session_state.run_start_time = time.time()
        # Ограничение гипероптимизации для слабых серверов
        if st.session_state.use_hyperopt and len(df) > 1500:
            st.warning("Гипероптимизация отключена из-за большого объема данных. Включите вручную только при необходимости.")
            st.session_state.use_hyperopt = False
            st.session_state.n_trials = min(st.session_state.n_trials, 10)

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
            inferred_freq, monthly_forced = adjust_daily_to_monthly(inferred_freq, df_indexed.index)
            
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
            
            inferred_freq, monthly_forced = adjust_daily_to_monthly(inferred_freq, df_indexed.index)
            
            df_sorted = normalize_month_start(df_sorted, time_col, inferred_freq)
            # Создаем TimeSeries с правильной частотой
            try:
                if len(df_sorted) >= 3:
                    series = safe_timeseries_from_df(
                        df_sorted,
                        time_col=time_col,
                        value_col=value_col,
                        freq=inferred_freq if inferred_freq else None,
                        label="расчётного ряда",
                    ).astype(np.float32)
                    
                    # Убедимся, что частота установлена корректно
                    if not hasattr(series, 'freq') or series.freq is None:
                        if inferred_freq:
                            # Пересоздаем ряд с определенной частотой
                            series = TimeSeries.from_times_and_values(
                                series.time_index,
                                series.values(),
                                freq=inferred_freq,
                                fill_missing_dates=True
                            )
                else:
                    st.error(f"Недостаточно данных для анализа. Требуется минимум 3 строки, а у вас {len(df_sorted)}.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Ошибка при создании временного ряда: {e}")
                st.stop()
            
            # FIX: Add a strict check for minimum training size to prevent IndexError
            min_train_size = 10
            max_val_size = len(series) - min_train_size
            if max_val_size <= 0:
                st.error(f"Ошибка: Недостаточно данных для обучения. Требуется минимум {min_train_size + 1} точек данных.")
                st.stop()

            # Валидируем на большем отрезке, чем horizon, если данных хватает
            suggested_val = max(n_forecast, max(5, int(len(series) * 0.2)))
            if max_val_size < n_forecast:
                st.error(f"Ошибка: Недостаточно данных для обучения. После выделения {n_forecast} точек под валидацию остаётся только {len(series) - n_forecast}, требуется минимум {min_train_size}. Уменьшите срок прогноза или загрузите больше данных.")
                st.stop()
            val_size = min(suggested_val, max_val_size)

            train, val = series[:-val_size], series[-val_size:]
            st.session_state.val_size = val_size
            
            future_covariates = None
            if extra_cols:
                # Создаем копию датафрейма для экзогенных переменных
                exog_df = df_sorted[[time_col] + extra_cols].copy()
                
                # Обрабатываем каждую экзогенную переменную
                for col in extra_cols:
                    # Удаляем нечисловые символы и преобразуем в числа
                    if exog_df[col].dtype == 'object':
                        exog_df[col] = exog_df[col].astype(str).str.replace(',', '.', regex=False)
                        exog_df[col] = exog_df[col].str.replace(r'[^\d.-]', '', regex=True)
                    
                    # Преобразуем в числовой формат
                    exog_df[col] = pd.to_numeric(exog_df[col], errors='coerce')
                    
                    # Заполняем пропущенные значения
                    exog_df[col].ffill(inplace=True)
                    exog_df[col].bfill(inplace=True)
                    exog_df[col].fillna(0, inplace=True)  # Оставшиеся NaN заполняем нулями
                
                try:
                    # Создаем TimeSeries для экзогенных переменных
                    future_covariates = TimeSeries.from_dataframe(
                        exog_df,
                        time_col=time_col,
                        fill_missing_dates=True,
                        freq=inferred_freq if inferred_freq else None
                    ).astype(np.float32)
                    
                    # Проверяем, что временные индексы совпадают
                    if not series.time_index.equals(future_covariates.time_index):
                        # Если индексы не совпадают, выравниваем их
                        common_time_index = series.time_index.intersection(future_covariates.time_index)
                        if len(common_time_index) == 0:
                            raise ValueError("Временные индексы основного ряда и экзогенных переменных не совпадают.")
                        
                        # Обрезаем оба ряда до общего временного диапазона
                        series = series.slice_intersect(future_covariates)
                        future_covariates = future_covariates.slice_intersect(series)
                    
                    # Проверяем размерности
                    if future_covariates.n_components != len(extra_cols):
                        raise ValueError(f"Ошибка размерности: ожидалось {len(extra_cols)} экзогенных переменных, получено {future_covariates.n_components}")
                    
                except Exception as e:
                    st.error(f"Ошибка при создании экзогенных переменных: {e}")
                    st.warning("Модели будут обучены без учета экзогенных переменных.")
                    future_covariates = None
                    extra_cols = []
                
                # Дополняем экзогенные переменные для периода прогноза (используем последние значения)
                if future_covariates is not None and len(future_covariates) < len(series) + n_forecast:
                    try:
                        from utils import _get_ts_values_and_index
                        last_values, last_index = _get_ts_values_and_index(future_covariates)
                        last_vals = last_values[-1] if len(last_values.shape) == 1 else last_values[-1, :]
                        
                        # Создаем дополнительные даты на основе частоты
                        if inferred_freq:
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
                        
                    except Exception as e:
                        st.warning(f"Не удалось дополнить экзогенные переменные для прогноза: {e}")
                        future_covariates = None
                
                # Если не удалось создать future_covariates, сбрасываем экзогенные переменные
                if future_covariates is None:
                    extra_cols = []
                    future_covariates = None

            # Выбор моделей согласно требованиям
            # Базовые модели (работают без экзогенных): ExponentialSmoothing, LinearRegression, Prophet, AutoARIMA, LightGBM, Theta, CatBoost, N-HiTS, TCN
            # Сложные модели (только с экзогенными): FFT, N-BEATS и другие
            base_models = ["ExponentialSmoothing", "LinearRegression", "Prophet", "AutoARIMA", "LightGBM", "Theta", "N-HiTS", "TCN"]
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
            
            # Выводим индикатор прогресса как оверлей по центру экрана
            overlay_placeholder = st.empty()

            def render_overlay(title: str, step_text: str, progress: float):
                percent = int(progress * 100)
                overlay_placeholder.markdown(
                    f"""
                    <div style="
                        position: fixed; inset: 0;
                        background: rgba(0, 0, 0, 0.55);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        z-index: 9999;">
                        <div style="
                            background: #0b1221;
                            color: #e5e7eb;
                            padding: 22px 26px;
                            border-radius: 18px;
                            width: min(420px, 90%);
                            box-shadow: 0 20px 60px rgba(0,0,0,0.45);
                            font-family: 'Inter', system-ui, -apple-system, sans-serif;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                                <span style="font-size: 18px; font-weight: 700;">{title}</span>
                                <span style="font-size: 14px; opacity: 0.8;">{percent}%</span>
                            </div>
                            <div style="font-size: 14px; margin-bottom: 12px; line-height: 1.5;">{step_text}</div>
                            <div style="background: rgba(255,255,255,0.08); border-radius: 999px; height: 12px; overflow: hidden;">
                                <div style="
                                    width: {percent}%;
                                    height: 100%;
                                    background: linear-gradient(90deg, #22d3ee, #6366f1);
                                    transition: width 180ms ease-out;"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            render_overlay("Начинаем обучение моделей", "⏳ Подготовка к обучению...", 0)
            
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
                    render_overlay(
                        "Подбор гиперпараметров",
                        f"🔍 {name}: поиск лучших настроек ({current_step}/{total_steps})",
                        progress,
                    )
                    best_params, opt_error = optimize_hyperparameters(
                        model_name=name,
                        train_series=train,
                        val_series=val,
                        forecast_horizon=len(val),
                        future_covariates=future_covariates,
                        n_trials=st.session_state.n_trials,
                        metric=st.session_state.ranking_metric.lower(),
                        season_length=st.session_state.season_period,
                    )
                    if opt_error:
                        st.warning(f"Не удалось оптимизировать {name}: {opt_error}. Используются параметры по умолчанию.")
                    else:
                        best_params_dict[name] = best_params
                
                # Обучение модели
                current_step += 1
                progress = current_step / total_steps
                render_overlay(
                    "Обучение моделей",
                    f"🚀 {name}: запуск обучения ({current_step}/{total_steps})",
                    progress,
                )
                
                forecast, model, error = train_model(
                    model_name=name, 
                    train_series=train,
                    forecast_horizon=len(val), 
                    future_covariates=future_covariates,
                    model_params=best_params if best_params else None,
                    season_length=st.session_state.season_period,
                )

                if error or forecast is None:
                    results_list.append({"Модель": name, "MAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "MSE": np.nan, "R2": np.nan, "Гиперпараметры": error or "Неизвестная ошибка"})
                    continue

                # Получаем значения для проверки нулей
                # Считаем метрики, выравнивая ряды по минимальной длине и убирая NaN
                from utils import _get_ts_values_and_index
                val_values, _ = _get_ts_values_and_index(val)
                fc_values, _ = _get_ts_values_and_index(forecast)
                min_len = min(len(val_values), len(fc_values))
                if min_len == 0:
                    mape_score = mae_score = r2_score_val = rmse_score = mse_score = np.nan
                else:
                    v_arr = val_values[:min_len].astype(float)
                    f_arr = fc_values[:min_len].astype(float)
                    mask = np.isfinite(v_arr) & np.isfinite(f_arr)
                    v_arr = v_arr[mask]
                    f_arr = f_arr[mask]
                    if len(v_arr) == 0:
                        mape_score = mae_score = r2_score_val = rmse_score = mse_score = np.nan
                    else:
                        # простые numpy-метрики
                        mae_score = float(np.mean(np.abs(v_arr - f_arr)))
                        mse_score = float(np.mean((v_arr - f_arr) ** 2))
                        rmse_score = float(np.sqrt(mse_score))
                        if 0 in v_arr:
                            mape_score = np.nan
                        else:
                            mape_score = float(np.mean(np.abs((v_arr - f_arr) / v_arr)) * 100)
                        var = np.var(v_arr)
                        if var == 0:
                            r2_score_val = np.nan
                        else:
                            ss_res = np.sum((v_arr - f_arr) ** 2)
                            ss_tot = np.sum((v_arr - np.mean(v_arr)) ** 2)
                            r2_score_val = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
                
                # Сохраняем информацию о гиперпараметрах
                if best_params:
                    params_str = f"Оптимизированные: {best_params}"
                else:
                    try:
                        params = model.model_params if hasattr(model, 'model_params') else getattr(model, 'model', {}).get_params() if hasattr(model, 'model') else {}
                        params_str = str(params) if params else "По умолчанию"
                    except:
                        params_str = "По умолчанию"
                
                results_list.append({"Модель": name, "MAPE": mape_score, "MAE": mae_score, "RMSE": rmse_score, "MSE": mse_score, "R2": r2_score_val, "Гиперпараметры": params_str})
                forecasts[name] = forecast
                trained_models[name] = model
            
            overlay_placeholder.empty()  # Убираем оверлей после завершения
            st.success("✅ Обучение всех моделей завершено!")

            if not results_list:
                overlay_placeholder.empty()
                st.error("Ни одна модель не смогла быть обучена."); st.stop()

            results_df = pd.DataFrame(results_list).set_index("Модель")
            # Если выбранная метрика вся NaN, пробуем подобрать другую, но не останавливаем выполнение
            metric_priority = ["MAE", "RMSE", "MAPE", "MSE"]
            non_nan_metrics = [m for m in metric_priority if m in results_df.columns and results_df[m].notna().any()]
            if not non_nan_metrics:
                st.warning("Ни одна модель не рассчитала метрики (все значения NaN). Показаны результаты как есть. Ниже подробности по моделям.")
                st.dataframe(results_df.reset_index())  # выводим сырые результаты с ошибками
                st.session_state.ranking_metric = metric_priority[0]
            elif not results_df[st.session_state.ranking_metric].notna().any():
                fallback_metric = non_nan_metrics[0]
                st.warning(f"Метрика {st.session_state.ranking_metric} недоступна (все NaN). Используется {fallback_metric}.")
                st.session_state.ranking_metric = fallback_metric
            # Определяем, в каком порядке сортировать:
            # для R2 и RMSE по убыванию (больше лучше), для остальных по возрастанию.
            ascending_flag = False if st.session_state.ranking_metric == "R2" else True


            results_df = results_df.sort_values(
                by=st.session_state.ranking_metric,
                ascending=ascending_flag,
                na_position='last'
            )
            
            # Создаем финальный прогноз лучшей модели на весь период
            non_na_results = results_df.dropna(subset=[st.session_state.ranking_metric])
            if non_na_results.empty:
                st.error("Ни одна модель не рассчитала выбранную метрику (все значения NaN). Попробуйте другую метрику или другой горизонт.")
                st.stop()
            best_model_name = non_na_results.index[0]
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
                        model_params=model_params if model_params else None,
                        season_length=st.session_state.season_period,
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
            if "overlay_placeholder" in locals():
                overlay_placeholder.empty()
            st.error(f"Произошла критическая ошибка на этапе выполнения: {e}")
            st.exception(e)
            st.stop()

    results_df = st.session_state.battle_results
    forecasts = st.session_state.forecasts
    
    st.subheader(f"🏆 Таблица результатов (прогноз на {st.session_state.n_forecast} шагов)")
    st.markdown(f"Ранжирование по: **{st.session_state.ranking_metric}**")

    def highlight_best(s):
        is_min = s.name in ["MAE", "MAPE", "RMSE", "MSE"]
        best_val = s.min() if is_min else s.max()
        return ['background-color: #28a745' if v == best_val else '' for v in s]

    st.dataframe(
        results_df.style.apply(
            highlight_best,
            subset=["MAE", "MAPE", "RMSE", "MSE", "R2"]
        ).format(
            {"MAPE": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "MSE": "{:.4f}", "R2": "{:.4f}"},
            na_rep="-"
        )
    )

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

            # Ensure the time column is in datetime format
            time_col = st.session_state.time_col
            value_col = st.session_state.value_col
            
            # Make a copy to avoid modifying the original
            plot_df = df_for_plot[[time_col, value_col]].copy()
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
                plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors='coerce')
            
            # Set the time column as index
            plot_df = plot_df.set_index(time_col).sort_index()
            
            plot_freq, plot_monthly_forced = adjust_daily_to_monthly(plot_freq, plot_df.index)

            # Try to infer frequency if not provided
            if not plot_freq:
                try:
                    plot_freq = pd.infer_freq(plot_df.index)
                    if not plot_freq:  # If frequency couldn't be inferred
                        # Calculate median time difference
                        time_diffs = plot_df.index.to_series().diff().dropna()
                        if not time_diffs.empty:
                            median_diff = time_diffs.median()
                            if pd.Timedelta(days=27) <= median_diff <= pd.Timedelta(days=33):
                                plot_freq = 'M'  # Monthly
                            elif median_diff >= pd.Timedelta(days=80) and median_diff <= pd.Timedelta(days=100):
                                plot_freq = 'Q'  # Quarterly
                            elif median_diff >= pd.Timedelta(days=300) and median_diff <= pd.Timedelta(days=400):
                                plot_freq = 'A'  # Yearly
                            elif median_diff <= pd.Timedelta(hours=2):
                                plot_freq = 'H'  # Hourly
                            else:
                                plot_freq = 'D'  # Daily as fallback
                except Exception:
                    plot_freq = None
            
            # Create TimeSeries with inferred frequency
            plot_df_reset = normalize_month_start(plot_df.reset_index(), time_col, plot_freq)
            series_to_plot = safe_timeseries_from_df(
                plot_df_reset,
                time_col=time_col,
                value_col=value_col,
                freq=plot_freq,
                label="графика тестового прогноза",
            ).astype(np.float32)
            val_size_plot = st.session_state.get("val_size", st.session_state.n_forecast)
            train_plot, val_plot = series_to_plot[:-val_size_plot], series_to_plot[-val_size_plot:]
            
            selected_forecasts = {name: forecasts[name] for name in models_to_plot if name in forecasts}
            fig_test = plot_forecast(train_plot, val_plot, selected_forecasts)
            st.plotly_chart(fig_test, width='stretch')
    else:
        st.warning("Ни одна модель не смогла построить прогноз, график недоступен.")
    
    # График финального прогноза лучшей модели
    st.subheader("🎯 Финальный прогноз на нужный период (лучшая модель)")
    non_na_results = results_df.dropna(subset=[st.session_state.ranking_metric])
    if non_na_results.empty:
        st.warning("Финальный прогноз недоступен: нет моделей с рассчитанной выбранной метрикой.")
    else:
        best_model_name = non_na_results.index[0]
    if st.session_state.final_forecast is not None and non_na_results is not None and not non_na_results.empty:
        time_col = st.session_state.time_col
        value_col = st.session_state.value_col
        
        # Prepare the final data
        df_for_final = st.session_state.df.sort_values(by=time_col).copy()
        df_for_final[value_col] = pd.to_numeric(df_for_final[value_col], errors='coerce')
        df_for_final.dropna(subset=[value_col], inplace=True)
        
        # Make a copy to avoid modifying the original
        plot_df = df_for_final[[time_col, value_col]].copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(plot_df[time_col]):
            plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors='coerce')
        
        # Remove any rows with NaT in the time column
        plot_df = plot_df.dropna(subset=[time_col])
        
        # Set the time column as index and sort
        plot_df = plot_df.set_index(time_col).sort_index()
        
        # Try to infer frequency
        final_freq = pd.infer_freq(plot_df.index)
        
        if not final_freq and len(plot_df) > 1:
            # Calculate median time difference
            time_diffs = plot_df.index.to_series().diff().dropna()
            if not time_diffs.empty:
                median_diff = time_diffs.median()
                if pd.Timedelta(days=23) <= median_diff <= pd.Timedelta(days=33):
                    final_freq = 'M'  # Monthly
                elif pd.Timedelta(hours=11) <= median_diff <= pd.Timedelta(hours=13):
                    final_freq = 'H'  # Hourly
                elif pd.Timedelta(hours=23) <= median_diff <= pd.Timedelta(hours=25):
                    final_freq = 'D'  # Daily
                elif pd.Timedelta(days=80) <= median_diff <= pd.Timedelta(days=100):
                    final_freq = 'Q'  # Quarterly
                elif pd.Timedelta(days=300) <= median_diff <= pd.Timedelta(days=400):
                    final_freq = 'A'  # Yearly

        final_freq, final_monthly_forced = adjust_daily_to_monthly(final_freq, plot_df.index)
        
        # Create TimeSeries with inferred frequency
        try:
            plot_df_reset = normalize_month_start(plot_df.reset_index(), time_col, final_freq)
            series_full = safe_timeseries_from_df(
                plot_df_reset,
                time_col=time_col,
                value_col=value_col,
                freq=final_freq if final_freq else None,
                label="финального прогноза",
            ).astype(np.float32)
        except Exception as e:
            # Fallback without frequency if there's an error
            st.warning(f"Could not set frequency for final forecast plot: {e}. Plotting without frequency.")
            series_full = TimeSeries.from_dataframe(
                plot_df_reset,
                time_col=time_col,
                value_cols=value_col,
                fill_missing_dates=False
            ).astype(np.float32)
        
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

    else:
        st.warning("Не удалось создать финальный прогноз для выгрузки.")
