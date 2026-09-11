"""
analyzer.py — парсинг Betaflight blackbox-логов (CSV) и эвристический расчёт PID
с учетом методологии Betaflight 4.4+ (гайды по тюнингу «по ощущениям» + blackbox-анализ:
Chris Rosser, UAV Tech, официальный PID Tuning Guide).

ИЗМЕНЕНИЯ ПО СРАВНЕНИЮ С ПРЕДЫДУЩЕЙ ВЕРСИЕЙ (см. пояснения в конце файла):
1. Введена нижняя граница анализируемых частот MIN_ANALYSIS_FREQ_HZ — раньше самый
   первый после-DC бин спектра (около 0.1–1 Гц) часто содержал огромную мощность из-за
   "утечки" низкочастотной составляющей самого манёвра (шаг setpoint), а не реального шума.
   Это одновременно (а) ломало масштаб графика FFT и (б) искажало oscillation_score/noise_score,
   так как total_power считался с учётом этого выброса.
2. Полоса "низких частот" разделена на две:
   - wobble_band (0.3–2.5 Гц)  — медленная раскачка, характерный признак избыточного I
     (I-term wobble / windup), которую предыдущая версия вообще не видела, т.к. полоса
     начиналась с 5 Гц.
   - p_osc_band (3–25 Гц) — классическая "P/D" раскачка (buzz на резких стиках).
   Это отдельные проблемы, требующие разных действий (I вниз vs P вниз/D вверх), раньше
   они не различались.
3. suggest_pid теперь реально трогает I (раньше i_mult ни разу не менялся, несмотря на то,
   что переменная существовала и клипалась — I оставался равен исходному значению всегда).
4. Диагностика P-vs-D при обнаруженной раскачке 3–25 Гц теперь смотрит на overshoot и
   bounce-back, чтобы не увеличивать D и одновременно уменьшать P вслепую по одному и тому же
   признаку (раньше это делалось всегда одновременно, что физически противоречиво: если
   реальная причина — слабый D, лишнее уменьшение P просто гасит отзывчивость, не решая проблему,
   и наоборот).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

AXES = ["roll", "pitch", "yaw"]

# Ниже этой частоты бины спектра исключаются из ЛЮБЫХ расчётов мощности (не только полос).
# Причина: даже после вычитания среднего, резкие изменения setpoint (сам манёвр) дают
# огромную "утечку" энергии в первые несколько бинов спектра, которая не является шумом
# или реальной низкочастотной раскачкой контура. Без этой отсечки total_power и графики
# доминируются нерелевантным пиком у 0 Гц.
MIN_ANALYSIS_FREQ_HZ = 2.0


@dataclass
class AxisMetrics:
    axis: str
    sample_rate_hz: float
    rms_error: float
    mean_abs_error: float
    overshoot_pct: float
    bounce_back_score: float  # Метрика отскока в конце маневра
    wobble_power: float       # Мощность в полосе 0.3-2.5 Гц (кандидат на I-term wobble)
    low_freq_power: float     # Мощность в полосе 3-25 Гц (кандидат на P/D раскачку)
    high_freq_power: float    # Мощность в полосе 80-300 Гц (шум/D)
    dominant_freq_hz: Optional[float]
    wobble_score: float        # wobble_power / total_power (>= MIN_ANALYSIS_FREQ_HZ)
    oscillation_score: float   # low_freq_power / total_power
    noise_score: float         # high_freq_power / total_power
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

    # ИСПРАВЛЕНО: раньше строка заголовка CSV признавалась началом данных только если в ней
    # ОДНОВРЕМЕННО встречались "time" и "vbat". Многие экспорты (в т.ч. синтетические демо-логи,
    # логи без записи vbat, некоторые версии blackbox_decode) не содержат "vbat" в заголовке —
    # в этом случае data_start оставался равен 0, и в pandas.read_csv попадала строка вида
    # "H Product:Blackbox..." как первая "строка данных", из-за чего парсер времени ниже падал
    # с "Не нашёл колонку времени", хотя колонка времени в файле была. Теперь ищем именно
    # CSV-заголовок: строку, где среди полей, разделённых запятой, есть поле, начинающееся с
    # "time" (после lower/strip), либо есть "loopiteration" — это не зависит от того, какие
    # ещё колонки логировались.
    for idx, line in enumerate(lines):
        if "," not in line:
            continue
        fields = [f.strip().strip('"').lower() for f in line.split(",")]
        if "loopiteration" in fields or any(f.startswith("time") for f in fields):
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
    """FFT спектр мощности. DC (среднее) вычитается, окно Ханна для снижения растекания
    спектра. Даже так, самые нижние бины (примерно ниже MIN_ANALYSIS_FREQ_HZ) всё ещё несут
    низкочастотную "утечку" формы самого манёвра, а не шум/раскачку — это фильтруется
    отдельно в _band_power / _total_power_above, а не здесь, чтобы freqs/power, отданные
    наружу для графика, оставались полными (график сам решает, что и как обрезать/отображать)."""
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


def _total_power_above(freqs: np.ndarray, power: np.ndarray, min_freq_hz: float = MIN_ANALYSIS_FREQ_HZ) -> float:
    """Суммарная мощность спектра выше min_freq_hz. Используется как знаменатель для всех
    *_score метрик, чтобы низкочастотная утечка манёвра не "разбавляла" (или наоборот не
    доминировала) долю реального шума/раскачки."""
    if len(freqs) == 0:
        return 0.0
    mask = freqs >= min_freq_hz
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def get_display_spectrum(freqs: Optional[np.ndarray], power: Optional[np.ndarray],
                          min_freq_hz: float = MIN_ANALYSIS_FREQ_HZ,
                          max_freq_hz: float = 400.0):
    """Готовит спектр для отображения на графике: обрезает частоты ниже min_freq_hz
    (там же, где живёт паразитный пик от самого манёвра, ломающий масштаб оси Y) и выше
    max_freq_hz. Возвращает (freqs_cropped, power_cropped, y_max_suggested).

    y_max_suggested — рекомендуемый верхний предел оси Y с запасом, посчитанный по
    99-му перцентилю ВИДИМОГО диапазона, а не по глобальному максимуму спектра (который
    в Plotly учитывается при автоскейле оси Y, даже если по оси X диапазон визуально обрезан).
    """
    if freqs is None or power is None or len(freqs) == 0:
        return np.array([]), np.array([]), 1.0

    mask = (freqs >= min_freq_hz) & (freqs <= max_freq_hz)
    f_disp = freqs[mask]
    p_disp = power[mask]

    if len(p_disp) == 0:
        return f_disp, p_disp, 1.0

    p99 = float(np.percentile(p_disp, 99))
    p_max = float(np.max(p_disp))
    # Если максимум не сильно выше 99-го перцентиля — используем его напрямую с небольшим
    # запасом. Если есть один резкий изолированный пик — не даём ему растянуть шкалу так,
    # что всё остальное превращается в плоскую линию у нуля.
    y_max = p_max * 1.1 if p_max <= p99 * 3 else p99 * 2.0
    if y_max <= 0:
        y_max = 1.0
    return f_disp, p_disp, y_max


def _estimate_overshoot(setpoint: np.ndarray, gyro: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    """Оценивает переброс (overshoot) и паразитный отскок (bounce-back) после завершения манёвра."""
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

            if end + int(sample_rate_hz * 0.08) < len(gyro):
                stop_segment = gyro[end: end + int(sample_rate_hz * 0.08)]
                if len(stop_segment) > 0:
                    dip = float(np.min(stop_segment))
                    bounce = max(0.0, (post_target - dip) / abs(step_size)) * 100
                    bounce_backs.append(bounce)
        else:
            peak = float(np.min(segment))
            overshoot = max(0.0, (post_target - peak) / abs(step_size)) * 100
            if end + int(sample_rate_hz * 0.08) < len(gyro):
                stop_segment = gyro[end: end + int(sample_rate_hz * 0.08)]
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

    if len(freqs) > 0 and (float(freqs[-1]) < 2 * MIN_ANALYSIS_FREQ_HZ):
        notes.append(
            "⚠️ Лог слишком короткий или частота логирования слишком низкая для надёжного "
            "разделения I-раскачки (0.3–2.5 Гц) и P/D-раскачки — используйте лог длиннее и с "
            "логированием от 1 кГц."
        )

    wobble_power = _band_power(freqs, power, 0.3, 2.5)
    low_power = _band_power(freqs, power, 3, 25)
    high_power = _band_power(freqs, power, 80, 300)
    total_power = _total_power_above(freqs, power) or 1.0

    dominant_freq = None
    above_min_mask = freqs >= MIN_ANALYSIS_FREQ_HZ if len(freqs) else np.array([], dtype=bool)
    if np.any(above_min_mask):
        candidate_freqs = freqs[above_min_mask]
        candidate_power = power[above_min_mask]
        peak_idx = int(np.argmax(candidate_power))
        dominant_freq = float(candidate_freqs[peak_idx])

    wobble_score = wobble_power / total_power if total_power else 0.0
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
        wobble_power=wobble_power,
        low_freq_power=low_power,
        high_freq_power=high_power,
        dominant_freq_hz=dominant_freq,
        wobble_score=wobble_score,
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
    Эвристика PID (Betaflight 4.4+ / общепринятая методология тюнинга по blackbox):

    - P: сила немедленной реакции на ошибку. Слишком высокий -> быстрые колебания (buzz),
      горячие моторы. Слишком низкий -> вялая, "ватная" реакция, просадка на резких стиках.
    - I: удержание угла против устойчивой внешней силы (ветер, смещённый центр тяжести).
      Слишком высокий -> медленная раскачка ~0.5-2 Гц (I-term wobble), подёргивание при
      армировании (I windup на земле). Слишком низкий -> дрейф/снос в затяжных манёврах.
    - D: демпфер, гасит перерегулирование от P. Слишком низкий -> отскок (bounce-back) в
      конце флипов/роллов, "звон" после просадки в свой же прополваш. Слишком высокий ->
      усиливает высокочастотный шум -> греет моторы, может провоцировать вялую медленную
      раскачку от собственного шума.

    Порядок диагностики: сначала различаем МЕДЛЕННУЮ раскачку (I) и БЫСТРУЮ раскачку (P/D),
    так как раньше это была одна и та же полоса частот и одна и та же (ошибочная) реакция.
    """
    notes: list[str] = []

    if m.current_pid is None:
        notes.append("Текущие PID не найдены в заголовке лога — используются дефолты (45, 80, 30).")
        p, i, d = 45.0, 80.0, 30.0
    else:
        p, i, d = m.current_pid

    p_mult, i_mult, d_mult = 1.0, 1.0, 1.0

    # 1. Медленная раскачка (0.3–2.5 Гц) — классический признак избыточного I (I-term wobble).
    #    Раньше эта полоса не выделялась отдельно и вообще не влияла на I.
    if m.wobble_score > 0.30:
        i_mult -= 0.08
        notes.append(
            "🌊 Обнаружена медленная раскачка (0.3–2.5 Гц) — похоже на избыточный I "
            "(I-term wobble). Рекомендуется немного снизить I."
        )
    elif m.wobble_score < 0.05 and m.mean_abs_error > 0:
        # Нет медленной раскачки — I скорее всего не завышен. Точно определить, что I
        # ЗАНИЖЕН (снос/дрейф в затяжных манёврах), по одному только логу без размеченных
        # манёвров ненадёжно, поэтому явного increase здесь намеренно нет — это должно
        # подтверждаться и субъективным ощущением полёта, не только логом.
        pass

    # 2. Быстрая раскачка (3–25 Гц) — либо P перетянут, либо D слишком слаб. Различаем по
    #    сопутствующим признакам, а не поднимаем/опускаем оба параметра вслепую одновременно.
    if m.oscillation_score > 0.35:
        freq_str = f" (~{m.dominant_freq_hz:.0f} Гц)" if m.dominant_freq_hz else ""
        if m.bounce_back_score > 8.0 and m.overshoot_pct <= 12:
            # Раскачка сопровождается отскоком на стопе, но не переброса при самом движении —
            # это сильнее указывает на нехватку демпфирования, а не на избыток P.
            d_mult += 0.12
            notes.append(
                f"⚡ Раскачка{freq_str} + выраженный отскок на стопе без сильного переброса — "
                "похоже на нехватку D, а не на избыток P. Рекомендуется поднять D."
            )
        elif m.overshoot_pct > 12:
            # Раскачка и явный переброс во время движения — P перетянут.
            p_mult -= 0.10
            notes.append(
                f"⚡ Раскачка{freq_str} с выраженным перебросом (overshoot) во время манёвра — "
                "похоже на избыток P. Рекомендуется снизить P."
            )
        else:
            # Оба признака выражены слабо, но сама раскачка есть — действуем мягко и в обе
            # стороны, как компромисс (это поведение старой версии, оставлено как fallback).
            p_mult -= 0.05
            d_mult += 0.05
            notes.append(
                f"⚡ Обнаружена раскачка{freq_str} без явного доминирующего признака (P или D). "
                "Небольшая коррекция в обе стороны: P немного вниз, D немного вверх."
            )
    elif m.oscillation_score < 0.08 and m.mean_abs_error > 0:
        p_mult += 0.03
        notes.append("💡 Контур стабилен, быстрых колебаний нет — траектория отслеживается чисто.")

    # 3. Отскок (bounce-back) в конце манёвра — даже если полоса 3-25 Гц не показала общей
    #    раскачки (отскок может быть коротким одиночным событием, не создающим устойчивый пик).
    if m.bounce_back_score > 8.0 and not (m.oscillation_score > 0.35):
        d_mult += 0.10
        notes.append(
            f"🔄 Зафиксирован отскок (bounce-back) в конце манёвра ~{m.bounce_back_score:.1f}%. "
            "Демпфирование (D) не успевает гасить инерцию — рекомендуется увеличить D."
        )

    # 4. Высокочастотный шум (риск перегрева моторов от завышенного D / плохой фильтрации).
    if m.noise_score > 0.30:
        d_mult -= 0.10
        notes.append(
            "🔊 Высокий уровень высокочастотного шума (80-300 Гц) в контуре ошибки. "
            "Завышенный D усиливает этот шум и греет моторы — рекомендуется уменьшить D "
            "и/или проверить пропеллеры и настройки Dynamic Notch/Gyro Lowpass (см. вкладку фильтров)."
        )

    # 5. Переброс (overshoot) без явной раскачки — тоже сигнал скорректировать P/D, но мягче,
    #    чем в пункте 2, где переброс уже учтён как основной признак.
    if m.overshoot_pct > 12 and not (m.oscillation_score > 0.35):
        p_mult -= 0.05
        d_mult += 0.05
        notes.append(
            f"🎯 Переброс (overshoot) составляет ~{m.overshoot_pct:.0f}% без выраженной "
            "раскачки. Небольшое снижение P и повышение D для более чёткой остановки."
        )

    # Ограничения безопасного шага (не даём алгоритму резко менять настройки за один раз).
    p_mult = float(np.clip(p_mult, 0.80, 1.20))
    i_mult = float(np.clip(i_mult, 0.85, 1.15))
    d_mult = float(np.clip(d_mult, 0.80, 1.25))

    new_p = round(p * p_mult)
    new_i = round(i * i_mult)
    new_d = round(d * d_mult)

    if not notes:
        notes.append("✅ Баланс P/I/D оптимален по доступным признакам. Явная коррекция не требуется.")

    notes.append(
        "ℹ️ Меняйте не больше одного-двух параметров за раз и перелетайте между правками — "
        "эвристика по одному логу не заменяет проверку в реальном полёте."
    )

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
