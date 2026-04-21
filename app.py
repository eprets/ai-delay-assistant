import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ИИ-ассистент для прогнозирования задержек",
    layout="wide"
)

# -------------------- БОКОВАЯ ПАНЕЛЬ (ТОЛЬКО ИНФОРМАЦИЯ) --------------------
with st.sidebar:
    st.header("ℹ️ О приложении")
    st.markdown(
        """
        **ИИ-ассистент для прогнозирования задержек задач в ИТ-проектах**
        
        Прототип анализирует загруженные данные, оценивает вероятность срыва сроков,
        определяет уровень риска и генерирует рекомендации на естественном языке.
        
        Рекомендации формируются с помощью **эмуляции языковой модели (LLM)**,
        что позволяет получить развёрнутые и практически полезные советы.
        """
    )
    st.divider()
    st.markdown("**Обязательные колонки входного файла:**")
    st.code(
        "\n".join([
            "task_id", "task_name", "stage_name", "planned_start", "planned_finish",
            "actual_start", "progress", "status", "priority", "effort_hours",
            "assignee", "workload", "dependencies_count", "requirement_changes", "blockers"
        ])
    )

# -------------------- КОНСТАНТЫ --------------------
REQUIRED_COLUMNS = [
    "task_id", "task_name", "stage_name", "planned_start", "planned_finish",
    "actual_start", "progress", "status", "priority", "effort_hours",
    "assignee", "workload", "dependencies_count", "requirement_changes", "blockers",
]

# -------------------- ФУНКЦИИ ЗАГРУЗКИ И ПРОВЕРКИ --------------------
def read_file(uploaded_file) -> pd.DataFrame:
    """Чтение CSV или Excel файла."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Поддерживаются только CSV и Excel файлы.")

def validate_columns(df: pd.DataFrame) -> list[str]:
    """Проверка наличия обязательных колонок."""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]

# -------------------- ПРЕОБРАЗОВАНИЯ ДАННЫХ --------------------
def priority_to_num(value: str) -> int:
    mapping = {
        "low": 1, "medium": 2, "high": 3, "critical": 4,
        "низкий": 1, "средний": 2, "высокий": 3, "критический": 4,
    }
    if pd.isna(value):
        return 2
    return mapping.get(str(value).strip().lower(), 2)

def blocker_to_num(value) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().lower()
    return 1 if text in {"1", "true", "yes", "да"} else 0

# -------------------- СКОРИНГОВАЯ МОДЕЛЬ (оценка риска) --------------------
def calculate_risk_probability(row: pd.Series) -> float:
    """Прозрачная скоринговая модель для учебного прототипа."""
    score = 0.0
    score += min(row["progress_gap"] / 100.0, 1.0) * 0.35
    score += ((row["priority_num"] - 1) / 3.0) * 0.10
    score += min(row["workload"] / 100.0, 1.0) * 0.15
    score += min(row["dependencies_count"], 5) / 5.0 * 0.10
    score += min(row["requirement_changes"], 5) / 5.0 * 0.10
    score += row["blockers_num"] * 0.15
    if row["days_to_deadline"] < 0:
        score += 0.20
    elif row["days_to_deadline"] <= 2:
        score += 0.10
    return float(np.clip(score, 0.0, 0.99))

def risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Низкий"
    if probability < 0.65:
        return "Средний"
    return "Высокий"

def risk_factors(row: pd.Series) -> str:
    factors = []
    if row["progress_gap"] > 20:
        factors.append("отставание по прогрессу")
    if row["workload"] >= 85:
        factors.append("высокая загрузка исполнителя")
    if row["dependencies_count"] >= 4:
        factors.append("большое число зависимостей")
    if row["requirement_changes"] >= 3:
        factors.append("частые изменения требований")
    if row["blockers_num"] == 1:
        factors.append("наличие блокирующих факторов")
    if row["days_to_deadline"] <= 2:
        factors.append("близкий срок завершения")
    return ", ".join(factors) if factors else "значимые факторы риска не выявлены"

# -------------------- ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ (ЭМУЛЯЦИЯ LLM) --------------------
def generate_ai_recommendation(task_context: dict) -> str:
    """
    Эмуляция языковой модели – генерация развёрнутых рекомендаций
    на основе выявленных факторов риска.
    """
    rec_parts = []
    if task_context.get("progress_gap", 0) > 20:
        rec_parts.append(
            "Рекомендую провести внеочередную встречу с исполнителем для выяснения причин "
            "отставания и пересмотра оценки оставшихся работ."
        )
    if task_context.get("workload", 0) >= 85:
        rec_parts.append(
            "Обратите внимание на критическую загрузку исполнителя. Рассмотрите возможность "
            "привлечения дополнительного ресурса или переноса некритичных задач."
        )
    if task_context.get("dependencies_count", 0) >= 4:
        rec_parts.append(
            "Высокое число зависимостей увеличивает неопределённость. Проверьте статусы "
            "блокирующих задач и синхронизируйте планы с их владельцами."
        )
    if task_context.get("requirement_changes", 0) >= 3:
        rec_parts.append(
            "Частые изменения требований создают риски для сроков. Рекомендую инициировать "
            "процедуру фиксации требований (requirements freeze) и оценить дополнительное время."
        )
    if task_context.get("blockers_num", 0) == 1:
        rec_parts.append(
            "Наличие блокера требует немедленного вмешательства. Назначьте ответственного "
            "за снятие блокировки и отслеживайте прогресс ежедневно."
        )
    if task_context.get("days_to_deadline", 10) <= 2 and task_context.get("delay_probability", 0) >= 0.65:
        rec_parts.append(
            "Крайний срок приближается, а риск высок. Рекомендую подготовить план 'Б' "
            "(упрощение функционала, привлечение дополнительных сил) и согласовать возможный "
            "перенос срока с заказчиком."
        )

    if not rec_parts:
        return "По текущим данным задача находится в допустимой зоне риска. Продолжайте мониторинг согласно утверждённому плану."
    return " ".join(rec_parts)

def recommendation(row: pd.Series) -> str:
    context = {
        "progress_gap": row.get("progress_gap", 0),
        "workload": row.get("workload", 0),
        "dependencies_count": row.get("dependencies_count", 0),
        "requirement_changes": row.get("requirement_changes", 0),
        "blockers_num": row.get("blockers_num", 0),
        "days_to_deadline": row.get("days_to_deadline", 10),
        "delay_probability": row.get("delay_probability", 0.0)
    }
    return generate_ai_recommendation(context)

# -------------------- ПОДГОТОВКА ДАННЫХ --------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["planned_start"] = pd.to_datetime(result["planned_start"], errors="coerce")
    result["planned_finish"] = pd.to_datetime(result["planned_finish"], errors="coerce")
    result["actual_start"] = pd.to_datetime(result["actual_start"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    result["planned_start"].fillna(today, inplace=True)
    result["planned_finish"].fillna(today + pd.Timedelta(days=7), inplace=True)

    numeric_cols = ["progress", "effort_hours", "workload", "dependencies_count", "requirement_changes"]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    result["priority_num"] = result["priority"].apply(priority_to_num)
    result["blockers_num"] = result["blockers"].apply(blocker_to_num)

    result["planned_duration_days"] = (
        (result["planned_finish"] - result["planned_start"]).dt.days
    ).clip(lower=1)

    result["days_elapsed"] = (
        (today - result["planned_start"]).dt.days
    ).clip(lower=0)

    result["expected_progress"] = (
        (result["days_elapsed"] / result["planned_duration_days"]) * 100
    ).clip(lower=0, upper=100)

    result["progress_gap"] = (result["expected_progress"] - result["progress"]).clip(lower=0)
    result["days_to_deadline"] = (result["planned_finish"] - today).dt.days

    result["delay_probability"] = result.apply(calculate_risk_probability, axis=1)
    result["risk_level"] = result["delay_probability"].apply(risk_level)
    result["risk_factors"] = result.apply(risk_factors, axis=1)
    result["recommendation"] = result.apply(recommendation, axis=1)

    return result

def make_result_table(df: pd.DataFrame) -> pd.DataFrame:
    output = df[[
        "task_id", "task_name", "stage_name", "assignee", "progress",
        "delay_probability", "risk_level", "risk_factors", "recommendation"
    ]].copy()

    output["delay_probability"] = (output["delay_probability"] * 100).round(1).astype(str) + "%"
    def format_risk_level(value):
        if value == "Высокий":
            return "🔴 Высокий"
        if value == "Средний":
            return "🟡 Средний"
        return "🟢 Низкий"
    output["risk_level"] = output["risk_level"].apply(format_risk_level)
    return output

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analysis_results")
    return output.getvalue()

# -------------------- ИНТЕРФЕЙС STREAMLIT --------------------
st.title("🤖 ИИ-ассистент для прогнозирования задержек в ИТ-проектах")
st.write(
    "Прототип анализирует данные задач проекта, оценивает вероятность задержки, "
    "определяет уровень риска и формирует рекомендации с помощью эмуляции ИИ."
)

uploaded_file = st.file_uploader(
    "Загрузите CSV или Excel файл с данными проекта",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        source_df = read_file(uploaded_file)
        st.subheader("📋 Загруженные входные данные")
        st.dataframe(source_df, use_container_width=True)

        missing_cols = validate_columns(source_df)
        if missing_cols:
            st.error("В файле отсутствуют обязательные колонки: " + ", ".join(missing_cols))
        else:
            if st.button("🚀 Запустить анализ", type="primary"):
                with st.spinner("Идёт анализ данных и генерация рекомендаций..."):
                    prepared_df = prepare_data(source_df)
                    result_df = make_result_table(prepared_df)

                total_tasks = len(prepared_df)
                high_risk_count = (prepared_df["risk_level"] == "Высокий").sum()
                medium_risk_count = (prepared_df["risk_level"] == "Средний").sum()
                low_risk_count = (prepared_df["risk_level"] == "Низкий").sum()

                st.markdown(
                    """
                    <style>
                    .metric-card {
                        border-radius: 14px;
                        padding: 18px 16px;
                        text-align: center;
                        color: white;
                        font-family: sans-serif;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        margin-bottom: 8px;
                    }
                    .metric-title { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
                    .metric-value { font-size: 28px; font-weight: 700; }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""<div class="metric-card" style="background-color:#3b82f6;">
                        <div class="metric-title">Всего задач</div><div class="metric-value">{total_tasks}</div></div>""",
                        unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""<div class="metric-card" style="background-color:#ef4444;">
                        <div class="metric-title">🔴 Высокий риск</div><div class="metric-value">{high_risk_count}</div></div>""",
                        unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""<div class="metric-card" style="background-color:#f59e0b;">
                        <div class="metric-title">🟡 Средний риск</div><div class="metric-value">{medium_risk_count}</div></div>""",
                        unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""<div class="metric-card" style="background-color:#10b981;">
                        <div class="metric-title">🟢 Низкий риск</div><div class="metric-value">{low_risk_count}</div></div>""",
                        unsafe_allow_html=True)

                st.subheader("📊 Результаты анализа")
                st.dataframe(result_df, use_container_width=True)

                excel_bytes = to_excel_bytes(result_df)
                st.download_button(
                    label="📥 Скачать отчёт Excel",
                    data=excel_bytes,
                    file_name="analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.subheader("📈 Распределение задач по уровню риска")
                risk_counts = prepared_df["risk_level"].value_counts()
                st.bar_chart(risk_counts)

                col_left, col_right = st.columns(2)
                with col_left:
                    st.subheader("📉 Гистограмма вероятности задержки")
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.hist(prepared_df["delay_probability"] * 100, bins=8, color='skyblue', edgecolor='black')
                    ax_hist.set_xlabel("Вероятность задержки, %")
                    ax_hist.set_ylabel("Количество задач")
                    st.pyplot(fig_hist)

                with col_right:
                    st.subheader("🥧 Круговая диаграмма уровней риска")
                    pie_counts = prepared_df["risk_level"].value_counts()
                    fig_pie, ax_pie = plt.subplots()
                    ax_pie.pie(pie_counts, labels=pie_counts.index, autopct="%1.1f%%", startangle=90)
                    ax_pie.axis('equal')
                    st.pyplot(fig_pie)

                st.subheader("⚠️ Задачи высокого риска")
                high_risk = result_df[result_df["risk_level"].str.contains("Высокий")]
                if high_risk.empty:
                    st.success("Задачи высокого риска не выявлены.")
                else:
                    st.error("Обнаружены задачи с высоким риском задержки")
                    st.dataframe(high_risk, use_container_width=True)

    except Exception as exc:
        st.error(f"Ошибка обработки файла: {exc}")
else:
    st.info("👆 Для начала работы загрузите файл с данными проекта.")