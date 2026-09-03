"""
app.py — Веб-интерфейс (Streamlit) для анализа PID-регуляторов FPV-дрона по логам Betaflight.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analyzer import (
    load_log,
    analyze_axis,
    generate_demo_log,
    AXES,
)

st.set_page_config(
    page_title="FPV PID Autotuner (Betaflight)",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 Автоанализ и тюнинг PID по логам Betaflight")
st.markdown(
    "Загрузите `.bbl`/`.bfl` файл лога Blackbox или воспользуйтесь **демо-режимом**, "
    "чтобы оценить качество настройки дрона и получить рекомендации по PID."
)

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("Источник данных")
uploaded_file = st.sidebar.file_uploader(
    "Выберите файл лога (.bbl, .bfl, .csv)", 
    type=["bbl", "bfl", "csv", "txt"]
)

use_demo = st.sidebar.checkbox("Использовать демо-данные", value=not uploaded_file)

file_bytes = None
if uploaded_file is not None and not use_demo:
    file_bytes = uploaded_file.read()
elif use_demo:
    file_bytes = generate_demo_log()
    st.sidebar.info("Загружены синтетические демо-данные.")

st.sidebar.markdown("---")
st.sidebar.header("Параметры анализа")
show_raw_plots = st.sidebar.checkbox("Показать сырые графики сигналов", value=True)
show_fft = st.sidebar.checkbox("Показать FFT-спектр частот", value=True)

# --- ОСНОВНАЯ ЛОГИКА ---
if file_bytes is None:
    st.warning("👈 Пожалуйста, загрузите файл лога в боковой панели или включите демо-режим.")
    st.stop()

try:
    with st.spinner("Парсим лог-файл и вычисляем метрики..."):
        df, headers, sample_rate_hz = load_log(file_bytes)
except Exception as e:
    st.error(f"Ошибка при чтении лога: {e}")
    st.stop()

# Вывод информации о прошивке / дроне из заголовков
if headers:
    with st.expander("ℹ️ Информация о логе и прошивке (Headers)"):
        cols = st.columns(3)
        i = 0
        for k, v in headers.items():
            cols[i % 3].text(f"{k}: {v}")
            i += 1

st.success(
    f"Лог успешно обработан! Частота дискретизации: **{sample_rate_hz:.1f} Гц**, "
    f"строк данных: **{len(df):,}**"
)

# Анализируем оси
axis_metrics = {}
for idx, axis in enumerate(AXES):
    metrics = analyze_axis(df, headers, idx, axis, sample_rate_hz)
    if metrics:
        axis_metrics[axis] = metrics

if not axis_metrics:
    st.error("Не удалось найти данные ни по одной оси (Roll, Pitch, Yaw).")
    st.stop()

# --- СВОДНАЯ ТАБЛИЦА РЕКОМЕНДАЦИЙ ---
st.markdown("### 📊 Сводка и рекомендации по PID")

summary_data = []
for axis, m in axis_metrics.items():
    cur = f"{m.current_pid[0]}, {m.current_pid[1]}, {m.current_pid[2]}" if m.current_pid else "Не найдены"
    sug = f"{m.suggested_pid[0]}, {m.suggested_pid[1]}, {m.suggested_pid[2]}" if m.suggested_pid else "—"
    
    summary_data.append({
        "Ось": axis.capitalize(),
        "Текущие PID (P, I, D)": cur,
        "Рекомендуемые PID": sug,
        "RMS Ошибки": f"{m.rms_error:.2f}",
        "Переброс (%)": f"{m.overshoot_pct:.1f}%",
        "Шум ( ВЧ )": f"{m.noise_score * 100:.1f}%",
    })

summary_df = pd.DataFrame(summary_data)
st.table(summary_df)

# Графики сравнения текущих и рекомендуемых P
bar_fig = go.Figure()
axes_names = [a.capitalize() for a in axis_metrics.keys()]
cur_p_vals = [m.current_pid[0] if m.current_pid else 45 for m in axis_metrics.values()]
sug_p_vals = [m.suggested_pid[0] if m.suggested_pid else 45 for m in axis_metrics.values()]

bar_fig.add_trace(go.Bar(name='Текущий P', x=axes_names, y=cur_p_vals, marker_color='indianred'))
bar_fig.add_trace(go.Bar(name='Рекомендуемый P', x=axes_names, y=sug_p_vals, marker_color='lightsalmon'))
bar_fig.update_layout(barmode='group', title="Сравнение параметра P (Текущий vs Рекомендуемый)", yaxis_title="Значение P")

# Добавлен уникальный key для предотвращения ошибки StreamlitDuplicateElementId
st.plotly_chart(bar_fig, use_container_width=True, key="summary_p_comparison_bar_chart")

# --- ДЕТАЛЬНЫЙ РАЗБОР ПО ОСЯМ ---
st.markdown("---")
st.markdown("### 🔍 Подробный разбор по осям")

tabs = st.tabs([a.capitalize() for a in axis_metrics.keys()])

for tab, (axis, m) in zip(tabs, axis_metrics.items()):
    with tab:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"#### Ось: {axis.upper()}")
            st.metric("RMS ошибки слежения", f"{m.rms_error:.2f}")
            st.metric("Средняя абс. ошибка", f"{m.mean_abs_error:.2f}")
            st.metric("Оценка переброса", f"{m.overshoot_pct:.1f}%")
            if m.dominant_freq_hz:
                st.metric("Доминирующая частота колебаний", f"{m.dominant_freq_hz:.1f} Гц")
            
            st.markdown("##### 💡 Выводы и советы:")
            for note in m.notes:
                st.info(f"• {note}")
                
            if m.current_pid and m.suggested_pid:
                st.markdown("##### ⚙️ Команда для CLI (Betaflight):")
                p_c, i_c, d_c = m.suggested_pid
                # Пример для pitch/roll/yaw в cli
                axis_idx_map = {"roll": 0, "pitch": 1, "yaw": 2}
                idx_num = axis_idx_map.get(axis, 0)
                st.code(f"set pid_{axis} = {int(p_c)},{int(i_c)},{int(d_c)}\nsave", language="text")

        with col2:
            if show_raw_plots and m.time_s is not None and m.gyro is not None:
                fig_time = go.Figure()
                # Ограничим точки для быстроты рендеринга, если точек слишком много
                step = max(1, len(m.time_s) // 3000)
                
                fig_time.add_trace(go.Scatter(
                    x=m.time_s[::step], y=m.setpoint[::step], 
                    name="Setpoint (Команда)", line=dict(color="orange", width=1.5)
                ))
                fig_time.add_trace(go.Scatter(
                    x=m.time_s[::step], y=m.gyro[::step], 
                    name="Gyro (Факт)", line=dict(color="dodgerblue", width=1)
                ))
                fig_time.update_layout(
                    title=f"Отработка задания по оси {axis.upper()}",
                    xaxis_title="Время (с)",
                    yaxis_title="Градусы/сек",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300
                )
                # Уникальный key для временного графика каждой оси
                st.plotly_chart(fig_time, use_container_width=True, key=f"time_chart_{axis}")

            if show_fft and m.freqs is not None and len(m.freqs) > 0:
                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(
                    x=m.freqs, y=m.power,
                    name="FFT ошибки", line=dict(color="mediumpurple", width=1.5)
                ))
                fig_fft.update_layout(
                    title=f"Спектр частот ошибки ({axis.upper()})",
                    xaxis_title="Частота (Гц)",
                    yaxis_title="Мощность",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=250,
                    xaxis=dict(range=[0, 200]) # Ограничим до 200 Гц для наглядности
                )
                # Уникальный key для FFT-графика каждой оси
                st.plotly_chart(fig_fft, use_container_width=True, key=f"fft_chart_{axis}")

st.markdown("---")
st.caption("FPV PID Autotuner • Построено на Streamlit, Pandas и Plotly")