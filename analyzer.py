"""
analyzer.py — парсинг Betaflight blackbox-логов (CSV) и эвристический расчёт PID
с учетом методологии Betaflight 4.4+ (по гайдам Криса Россера и практикам форумного тюнинга).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

AXES = ["roll", "pitch", "yaw"]


@dataclass
class AxisMetrics:
    axis: str
    sample_rate_hz: float
    rms_error: float
    mean_abs_error: float
    overshoot_pct: float
    bounce_back_score: float  # Новая метрика отскока в конце маневра
    low_freq_power: float
    high_freq_power: float
    dominant_freq_hz: Optional[float]
    oscillation_score: float
    noise_score: float
    current_pid: Optional[tuple] = None
    suggested_pid: Optional[tuple] = None
    notes: list = field(default_factory=list)
    freqs: Optional[np.ndarray] = None
    power: Optional[np.ndarray] = None
    time_s: Optional[np.ndarray] = None
    gyro: Optional[np.ndarray] = None
    setpoint: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def _find_dynamic_column(df: pd.DataFrame, kind: str, axis_index: int) -> Optional[str]:
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    
    if kind == "time":
        for pat in ["time (us)", "time(us)", "time"]:
            for i, cl in enumerate(cols_lower):
                if pat == cl:
                    return cols[i]
        return None

    if kind == "gyro":
        keywords = [
            f"gyroadc[{axis_index}]", f"gyrodata[{axis_index}]", f"gyro[{axis_index}]",
            f"gyrounfilt[{axis_index}]", f"gyro_{axis_index}", f"gyro[{axis_index},0]"
        ]
    elif kind == "setpoint":
        keywords = [
            f"setpoint[{axis_index}]", f"rccommand[{axis_index}]", f"rccommand[{axis_index}]".lower(),
            f"setpoint_{axis_index}", f"debug[{axis_index}]"
        ]
    else:
        keywords = []

    for kw in keywords:
        for i, cl in enumerate(cols_lower):
            if kw in cl:
                return cols[i]

    target_words = []
    if kind == "gyro":
        target_words = ["gyro", "gyroadc", "gyrounfilt"]
    elif kind == "setpoint":
        target_words = ["setpoint", "rccommand"]

    matching_cols = []
    for i, cl in enumerate(cols_lower):
        if any(w in cl for w in target_words):
            matching_cols.append(cols[i])

    if len(matching_cols) > axis_index:
        return matching_cols[axis_index]
    elif len(matching_cols) == 1:
        return matching_cols[0]

    return None


def parse_header(raw_text: str) -> dict:
    headers = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if line.startswith("H "):
            content = line[2:]
            if ":" in content:
                key, val = content.split(":", 1)
                headers[key.strip()] = val.strip()
        elif line and not line.startswith("H") and ("loopiteration" in line.lower() or "time" in line.lower()):
            break
    return headers


def _extract_current_pid(headers: dict, axis: str) -> Optional[tuple]:
    key_candidates = [f"{axis}PID", f"{axis}_pid", f"{axis.capitalize()}PID"]
    for key in key_candidates:
        if key in headers:
            parts = [p.strip() for p in headers[key].split(",")]
            try:
                nums = [float(p) for p in parts[:3]]
                if len(nums) == 3:
                    return tuple(nums)
            except ValueError:
                continue
    return None


def load_log(file_bytes: bytes) -> tuple[pd.DataFrame, dict, float]:
    text = file_bytes.decode("utf-8", errors="replace")
    headers = parse_header(text)

    lines = text.splitlines()
    data_start = 0
    
    for idx, line in enumerate(lines):
        low = line.lower()
        if "loopiteration" in low or ("time" in low and "vbat" in low):
            data_start = idx
            break

    csv_text = "\n".join(lines[data_start:])
    
    df = pd.read_csv(
        io.StringIO(csv_text), 
        skip_blank_lines=True, 
        engine="python", 
        on_bad_lines='skip'
    )
    df = _clean_columns(df)

    time_col = _find_dynamic_column(df, "time", 0)
    if time_col is None:
        raise ValueError(f"Не нашёл колонку времени в CSV. Доступные колонки: {list(df.columns)[:15]}")

    test_gyro = _find_dynamic_column(df, "gyro", 0)
    if test_gyro is None:
        raise ValueError(f"Не нашёл колонку гироскопа. Доступные колонки: {list(df.columns)}")

    t = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy()
    if len(t) < 10:
        raise ValueError("В логе слишком мало валидных строк данных.")

    dt = np.median(np.diff(t))
    sample_rate_hz = 1e6 / dt if dt > 0 else 1000.0

    return df, headers, sample_rate_hz


def _fft_bands(signal: np.ndarray, sample_rate_hz: float):
    n = len(signal)
    if n < 16:
        return np.array([]), np.array([])
    sig = signal - np.mean(signal)
    window = np.hanning(n)
    fft_vals = np.fft.rfft(sig * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    power = np.abs(fft_vals) ** 2
    return freqs, power


def _band_power(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    if len(freqs) == 0:
        return 0.0
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def _estimate_overshoot(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    """Оценивает переброс (overshoot) и паразитный отскок (bounce-back) после завершения маневра."""
    if len(setpoint) < int(sample_rate_hz * 0.1):
        return 0.0, 0.0

    sp_std = float(np.std(setpoint))
    if sp_std < 1e-6:
        return 0.0, 0.0

    d_setpoint = np.diff(setpoint)
    abs_d = np.abs(d_setpoint)
    threshold = max(float(np.percentile(abs_d, 99)), 0.15 * sp_std)
    if threshold <= 0:
        return 0.0, 0.0

    step_idxs = np.where(abs_d > threshold)[0]
    window = int(sample_rate_hz * 0.15)
    overshoots = []
    bounce_backs = []

    for idx in step_idxs:
        if idx < 1:
            continue
        pre_level = setpoint[idx]
        target_idx = min(idx + max(window // 3, 1), len(setpoint) - 1)
        post_target = setpoint[target_idx]
        step_size = post_target - pre_level
        if abs(step_size) < 0.1 * sp_std:
            continue

        end = min(idx + window, len(gyro) - 1)
        if end - idx < 3:
            continue
        segment = gyro[idx:end + 1]

        if step_size > 0:
            peak = float(np.max(segment))
            overshoot = max(0.0, (peak - post_target) / abs(step_size)) * 100
            
            # Поиск отскока в самом конце остановки (bounce-back)
            if end + int(sample_rate_hz * 0.08) < len(gyro):
                stop_segment = gyro[end : end + int(sample_rate_hz * 0.08)]
                if len(stop_segment) > 0:
                    dip = float(np.min(stop_segment))
                    bounce = max(0.0, (post_target - dip) / abs(step_size)) * 100
                    bounce_backs.append(bounce)
        else:
            peak = float(np.min(segment))
            overshoot = max(0.0, (post_target - peak) / abs(step_size)) * 100
            if end + int(sample_rate_hz * 0.08) < len(gyro):
                stop_segment = gyro[end : end + int(sample_rate_hz * 0.08)]
                if len(stop_segment) > 0:
                    spike = float(np.max(stop_segment))
                    bounce = max(0.0, (spike - post_target) / abs(step_size)) * 100
                    bounce_backs.append(bounce)

        overshoots.append(overshoot)

    med_overshoot = float(np.clip(np.median(overshoots), 0, 200)) if overshoots else 0.0
    med_bounce = float(np.clip(np.median(bounce_backs), 0, 200)) if bounce_backs else 0.0
    return med_overshoot, med_bounce


def analyze_axis(df: pd.DataFrame, headers: dict, axis_index: int, axis: str,
                 sample_rate_hz: float) -> Optional[AxisMetrics]:
    gyro_col = _find_dynamic_column(df, "gyro", axis_index)
    sp_col = _find_dynamic_column(df, "setpoint", axis_index)

    if gyro_col is None:
        return None

    notes: list[str] = []
    gyro = pd.to_numeric(df[gyro_col], errors="coerce").fillna(0).to_numpy()
    
    if len(gyro) == 0 or np.all(gyro == gyro[0]):
        notes.append("❌ Данные гироскопа статичны или отсутствуют для этой оси.")

    if sp_col is not None:
        setpoint = pd.to_numeric(df[sp_col], errors="coerce").fillna(0).to_numpy()
    else:
        setpoint = pd.Series(gyro).rolling(5, min_periods=1, center=True).mean().to_numpy()
        notes.append("⚠️ Колонка setpoint не найдена, оценка отработки заданий ограничена.")

    n = min(len(gyro), len(setpoint))
    gyro, setpoint = gyro[:n], setpoint[:n]
    error = setpoint - gyro
    time_s = np.arange(n) / sample_rate_hz

    rms_error = float(np.sqrt(np.mean(error ** 2)))
    mean_abs_error = float(np.mean(np.abs(error)))

    freqs, power = _fft_bands(error, sample_rate_hz)
    low_power = _band_power(freqs, power, 5, 30)
    high_power = _band_power(freqs, power, 80, 300)
    total_power = float(np.sum(power)) if len(power) else 1.0

    dominant_freq = None
    if len(power) > 1:
        peak_idx = int(np.argmax(power[1:]) + 1)
        dominant_freq = float(freqs[peak_idx])

    oscillation_score = low_power / total_power if total_power else 0.0
    noise_score = high_power / total_power if total_power else 0.0

    overshoot_pct, bounce_back_score = _estimate_overshoot(setpoint, gyro, sample_rate_hz)
    current_pid = _extract_current_pid(headers, axis)

    metrics = AxisMetrics(
        axis=axis,
        sample_rate_hz=sample_rate_hz,
        rms_error=rms_error,
        mean_abs_error=mean_abs_error,
        overshoot_pct=overshoot_pct,
        bounce_back_score=bounce_back_score,
        low_freq_power=low_power,
        high_freq_power=high_power,
        dominant_freq_hz=dominant_freq,
        oscillation_score=oscillation_score,
        noise_score=noise_score,
        current_pid=current_pid,
        notes=notes,
        freqs=freqs,
        power=power,
        time_s=time_s,
        gyro=gyro,
        setpoint=setpoint,
        error=error,
    )
    
    suggested_pid, suggest_notes = suggest_pid(metrics)
    metrics.suggested_pid = suggested_pid
    metrics.notes = notes + suggest_notes
    return metrics


def suggest_pid(m: AxisMetrics) -> tuple[Optional[tuple], list[str]]:
    """
    Эвристика PID с учетом современной методологии (Крис Россер / Betaflight 4.4+):
    - Баланс пружины (P) и демпфера/амортизатора (D).
    - Диагностика отскоков в конце маневров (bounce-back).
    - Разделение низкочастотных раскачек (слабый D или избыточный P) и ВЧ-шумов.
    """
    notes: list[str] = []

    if m.current_pid is None:
        notes.append("Текущие PID не найдены в заголовке лога — используются дефолты (45, 80, 30).")
        p, i, d = 45.0, 80.0, 30.0
    else:
        p, i, d = m.current_pid

    p_mult, i_mult, d_mult = 1.0, 1.0, 1.0

    # 1. Анализ низкочастотных колебаний (осцилляций)
    if m.oscillation_score > 0.35:
        p_mult -= 0.10
        d_mult += 0.08
        freq_str = f" (~{m.dominant_freq_hz:.0f} Гц)" if m.dominant_freq_hz else ""
        notes.append(
            f"⚡ Обнаружены низкочастотные колебания{freq_str}. "
            "Пружина (P) перенатянута или амортизатор (D) слишком слаб — рекомендуется снизить P и поднять D."
        )
    elif m.oscillation_score < 0.08 and m.mean_abs_error > 0:
        p_mult += 0.03
        notes.append("💡 Контур стабилен, колебаний нет — траектория отслеживается чисто.")

    # 2. Анализ паразитного отскока в конце роллов/флипов (классический симптом пункта #1 шпаргалки)
    if m.bounce_back_score > 8.0:
        d_mult += 0.10
        notes.append(
            f"🔄 Зафиксирован отскок (bounce-back) в конце маневра ~{m.bounce_back_score:.1f}%. "
            "Демпфирование (D) не успевает гасить инерцию — необходимо увеличить D."
        )

    # 3. Анализ высокочастотного шума (риск перегрева моторов)
    if m.noise_score > 0.30:
        d_mult -= 0.10
        notes.append("🔊 Высокий уровень высокочастотного шума в петле. Чрезмерное значение D может перегреть моторы — рекомендуется уменьшить D или проверить фильтры/пропеллеры.")

    # 4. Анализ переброса (overshoot)
    if m.overshoot_pct > 12:
        p_mult -= 0.05
        d_mult += 0.05
        notes.append(
            f"🎯 Переброс (overshoot) составляет ~{m.overshoot_pct:.0f}%. "
            "Дрон проскакивает целевую точку — слегка уменьшаем P и увеличиваем D для жесткой фиксации."
        )

    # Ограничения безопасного шага (не даем алгоритму резко менять настройки за один раз)
    p_mult = float(np.clip(p_mult, 0.80, 1.20))
    i_mult = float(np.clip(i_mult, 0.90, 1.10))
    d_mult = float(np.clip(d_mult, 0.80, 1.25))

    new_p = round(p * p_mult)
    new_i = round(i * i_mult)
    new_d = round(d * d_mult)

    if not notes:
        notes.append("✅ Баланс P/I/D оптимален. Корректировка не требуется.")

    return (new_p, new_i, new_d), notes


def generate_demo_log(seed: int = 42, duration_s: float = 6.0, sample_rate_hz: float = 1000.0) -> bytes:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    t_us = (np.arange(n) / sample_rate_hz * 1e6).astype(np.int64)

    header_lines = [
        "H Product:Blackbox flight data recorder by Nicholas Sherlock",
        "H rollPID:42,78,28",
        "H pitchPID:44,80,29",
        "H yawPID:60,90,0",
    ]

    cols = {"time (us)": t_us}
    for i, axis in enumerate(AXES):
        base = 200 * np.sign(np.sin(2 * np.pi * 0.3 * np.arange(n) / sample_rate_hz))
        setpoint = base + 40 * np.sin(2 * np.pi * 1.5 * np.arange(n) / sample_rate_hz)

        delay = int(0.02 * sample_rate_hz)
        gyro = np.zeros(n)
        gyro[delay:] = setpoint[:-delay] if delay > 0 else setpoint
        decay_phase = np.arange(n) % int(0.3 * sample_rate_hz)
        oscillation = 8 * np.exp(-0.02 * decay_phase) * np.sin(2 * np.pi * (18 + i * 4) * np.arange(n) / sample_rate_hz)
        noise = rng.normal(0, 3.0 if axis != "yaw" else 1.5, n)
        high_freq_noise = 2.0 * np.sin(2 * np.pi * 150 * np.arange(n) / sample_rate_hz)

        gyro = gyro + oscillation + noise + high_freq_noise

        cols[f"setpoint[{i}]"] = setpoint
        cols[f"gyroADC[{i}]"] = gyro
        cols[f"axisP[{i}]"] = rng.normal(0, 5, n)
        cols[f"axisI[{i}]"] = rng.normal(0, 2, n)
        cols[f"axisD[{i}]"] = rng.normal(0, 3, n)

    df = pd.DataFrame(cols)
    csv_body = df.to_csv(index=False)
    full_text = "\n".join(header_lines) + "\n" + csv_body
    return full_text.encode("utf-8")
