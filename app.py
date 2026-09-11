"""
app.py — Веб-интерфейс (Streamlit) для FPV Autotuner.
Интегрирован с продвинутым анализатором логов Betaflight 4.4+ (баланс P/D, отскоки, FFT-фильтры).
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from analyzer import load_log, analyze_axis, AXES
from filters import analyze_noise_and_filters

st.set_page_config(
    page_title="FPV PID & Filter Autotuner",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 Автоанализ PID и фильтров по логам Betaflight")
st.markdown(
    "Загрузите лог Blackbox (.bbl / .bfl), чтобы оценить точность отработки, "
    "найти частоты резонансов, отскоки в маневрах и получить рекомендации по PID и фильтрам."
)

st.sidebar.header("Источник данных")
uploaded_file = st.sidebar.file_uploader(
    "Выберите файл лога (.bbl, .bfl, .csv)", 
    type=["bbl", "bfl", "csv", "txt"]
)

file_bytes = uploaded_file.read() if uploaded_file is not None else None

st.sidebar.markdown("---")
st.sidebar.header("Параметры отображения")
show_raw_plots = st.sidebar.checkbox("Показать графики сигналов", value=True)
show_fft = st.sidebar.checkbox("Показать FFT-спектр частот", value=True)

if file_bytes is None:
    st.warning("👈 Пожалуйста, загрузите файл лога в боковой панели.")
    st.stop()

try:
    with st.spinner("Парсим лог-файл..."):
        df, headers, sample_rate_hz = load_log(file_bytes)
except Exception as e:
    st.error(f"Ошибка при чтении лога: {e}")
    st.stop()

axis_metrics = {}
for idx, axis in enumerate(AXES):
    metrics = analyze_axis(df, headers, idx, axis, sample_rate_hz)
    if metrics:
        axis_metrics[axis] = metrics

if not axis_metrics:
    st.error("Не удалось найти данные ни по одной оси.")
    st.stop()

# --- ВКЛАДКИ ИНТЕРФЕЙСА ---
tab_summary, tab_pids, tab_filters = st.tabs(["📊 Общая сводка", "⚙️ Тюнинг PID", "🛡️ Анализ шумов и фильтры"])

with tab_summary:
    st.markdown("### Сводная информация по полету")
    summary_data = []
    for axis, m in axis_metrics.items():
        cur = f"{int(m.current_pid[0])}, {int(m.current_pid[1])}, {int(m.current_pid[2])}" if m.current_pid else "—"
        sug = f"{int(m.suggested_pid[0])}, {int(m.suggested_pid[1])}, {int(m.suggested_pid[2])}" if m.suggested_pid else "—"
        summary_data.append({
            "Ось": axis.capitalize(),
            "Текущие PID": cur,
            "Рекомендуемые PID": sug,
            "RMS Ошибки": f"{m.rms_error:.2f}",
            "Переброс": f"{m.overshoot_pct:.1f}%",
            "Отскок (Bounce)": f"{m.bounce_back_score:.1f}%",
        })
    st.table(pd.DataFrame(summary_data))
    
    st.info(
        "💡 **Справка по методике тюнинга:** Рекомендации опираются на баланс пружины (P) и демпфера (D) "
        "по гайдам Криса Россера. Высокий процент отскока в конце маневра указывает на недостаток демпфирования (D), "
        "а низкочастотные осцилляции сигнализируют о необходимости снизить P или поднять D."
    )

with tab_pids:
    st.markdown("### Детальный разбор отработки PID по осям")
    sub_tabs = st.tabs([a.capitalize() for a in axis_metrics.keys()])
    for tab, (axis, m) in zip(sub_tabs, axis_metrics.items()):
        with tab:
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.metric("RMS ошибки", f"{m.rms_error:.2f}")
                st.metric("Переброс (Overshoot)", f"{m.overshoot_pct:.1f}%")
                st.metric("Отскок (Bounce-back)", f"{m.bounce_back_score:.1f}%")
                
                st.markdown("**Диагностика и советы:**")
                for note in m.notes:
                    st.info(f"• {note}")
                    
                if m.current_pid and m.suggested_pid:
                    p_c, i_c, d_c = m.suggested_pid
                    st.markdown("##### 💻 Команда для CLI:")
                    st.code(f"set pid_{axis} = {int(p_c)},{int(i_c)},{int(d_c)}\nsave", language="text")
            with col2:
                if show_raw_plots and m.time_s is not None and m.gyro is not None:
                    fig = go.Figure()
                    step = max(1, len(m.time_s) // 3000)
                    fig.add_trace(go.Scatter(x=m.time_s[::step], y=m.setpoint[::step], name="Setpoint", line=dict(color="orange", width=1.5)))
                    fig.add_trace(go.Scatter(x=m.time_s[::step], y=m.gyro[::step], name="Gyro", line=dict(color="dodgerblue", width=1.0)))
                    fig.update_layout(
                        title=f"Отработка задания ({axis.upper()})", 
                        xaxis_title="Время (с)",
                        yaxis_title="Градусы / сек",
                        height=320, 
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"time_pid_{axis}")

with tab_filters:
    st.markdown("### Анализ спектра шумов и рекомендации по фильтрации")
    filter_sub_tabs = st.tabs([a.capitalize() for a in axis_metrics.keys()])
    for tab, (axis, m) in zip(filter_sub_tabs, axis_metrics.items()):
        with tab:
            f_res = analyze_noise_and_filters(m.freqs, m.power)
            
            col1, col2 = st.columns([1, 1.5])
            with col1:
                st.markdown(f"#### Ось: {axis.upper()}")
                for rec in f_res.recommendations:
                    st.warning(f"• {rec}")
                if f_res.cli_commands:
                    st.markdown("##### ⚙️ Рекомендуемые команды фильтров для CLI:")
                    st.code("\n".join(f_res.cli_commands), language="text")
            with col2:
                if show_fft and m.freqs is not None and m.power is not None:
                    fig_fft = go.Figure()
                    fig_fft.add_trace(go.Scatter(x=m.freqs, y=m.power, name="Спектр шума", line=dict(color="mediumpurple")))
                    for peak in f_res.noise_peaks:
                        fig_fft.add_vline(x=peak, line_dash="dash", line_color="red", annotation_text=f"{peak:.0f}Hz")
                    fig_fft.update_layout(
                        title=f"FFT спектр шумов ({axis.upper()})",
                        xaxis_title="Частота (Гц)", 
                        yaxis_title="Мощность",
                        height=320, 
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis=dict(range=[0, 400])
                    )
                    st.plotly_chart(fig_fft, use_container_width=True, key=f"fft_filter_{axis}")

st.markdown("---")
st.caption("FPV PID & Filter Autotuner • Модульная архитектура (Betaflight 4.4+)")
