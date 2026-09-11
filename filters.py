"""
filters.py — Модуль анализа спектра шумов и формирования рекомендаций по фильтрам Betaflight
с учетом принципов Betaflight 4.4+ и гайдов Криса Россера (минимизация задержки, RPM-фильтры, подбор нотчей).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FilterAnalysisResult:
    noise_peaks: List[float] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    cli_commands: List[str] = field(default_factory=list)


def analyze_noise_and_filters(freqs: Optional[np.ndarray], power: Optional[np.ndarray]) -> FilterAnalysisResult:
    """
    Анализирует FFT спектр гироскопа с проверкой качества данных.

    Логика работы:
    1. Проверяет наличие и целостность входных данных.
    2. Обязательно отсекает диапазон ниже 50 Гц (где находятся движения стиков пилота и маневры дрона).
    3. Оценивает соотношение максимального пика к среднему уровню шума в рабочем диапазоне (50–400 Гц).
    4. Игнорирует мелкий фоновый шум и ищет только полноценные резонансы (>= 15-20% от максимума).
    5. Если обнаружены реальные резонансы рамы или моторов — классифицирует их по частотам
       и формирует точные команды для CLI Betaflight.
    """
    if freqs is None or power is None or len(freqs) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: массив частот или мощности пуст."]
        )

    noise_peaks: List[float] = []
    recommendations: List[str] = []
    cli_commands: List[str] = []

    # Шаг 1: Игнорируем зону 0-50 Гц (там живут движения стиков и пилотирование, а не шум моторов)
    valid_mask = freqs >= 50.0
    if not np.any(valid_mask):
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: частотная сетка лога слишком узкая для поиска резонансов (требуется диапазон >50 Гц)."]
        )

    f_valid = freqs[valid_mask]
    p_valid = power[valid_mask]

    if len(p_valid) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Отсутствуют данные гироскопа в диапазоне выше 50 Гц."]
        )

    max_p = np.max(p_valid)
    mean_p = np.mean(p_valid)

    # Шаг 2: Проверка на «плоский» или чистый лог
    if max_p <= mean_p * 2.0 or max_p < 0.5:
        recommendations.append(
            "⚠️ Лог в рабочей зоне (>50 Гц) выглядит относительно ровным или чистым (нет резких выраженных пиков резонанса)."
        )
        recommendations.append(
            "💡 Что проверить: убедитесь, что в конфигураторе Blackbox включен лог сырого гироскопа (pre-filter) "
            "и что в файле записан реальный активный полет с нагрузкой, а не просто висение или статика на столе."
        )
        return FilterAnalysisResult(
            noise_peaks=[],
            recommendations=recommendations,
            cli_commands=[]
        )

    # Шаг 3: Поиск реальных пиков вибраций с фильтрацией мелкого шума (требуем, чтобы пик был весомым)
    if max_p > mean_p * 3.5 and max_p > 500.0:
        window_size = max(5, int(len(f_valid) * 0.03))
        if window_size % 2 == 0:
            window_size += 1

        envelope = np.copy(p_valid)
        half_w = window_size // 2
        for i in range(len(p_valid)):
            start = max(0, i - half_w)
            end = min(len(p_valid), i + half_w + 1)
            envelope[i] = np.max(p_valid[start:end])

        env_mean = np.mean(envelope)
        # Жесткий порог: пик должен быть заметно выше фона И составлять не менее 15-20% от максимума
        env_threshold = max(env_mean * 2.5, max_p * 0.18)

        candidates = []
        for i in range(half_w, len(f_valid) - half_w):
            if f_valid[i] < 50.0:
                continue

            current_val = envelope[i]
            if current_val >= env_threshold:
                is_local_max = True
                for j in range(i - half_w, i + half_w + 1):
                    if envelope[j] > current_val:
                        is_local_max = False
                        break

                if is_local_max:
                    f_cand = float(f_valid[i])
                    if not candidates or all(abs(f_cand - existing) > 40.0 for existing in candidates):
                        candidates.append(f_cand)

        if candidates:
            scored_peaks = []
            for cf in candidates:
                idx = np.argmin(np.abs(f_valid - cf))
                scored_peaks.append((p_valid[idx], cf))

            scored_peaks.sort(key=lambda x: x[0], reverse=True)
            # Отбираем только те пики, которые реально «тянут» на полноценный мешающий шум (>= 20% от максимума)
            noise_peaks = sorted([item[1] for item in scored_peaks[:2] if item[0] >= max_p * 0.20])

    # Шаг 4: Формирование экспертных рекомендаций на основе найденных частот (методика Криса Россера)
    if noise_peaks:
        peaks_str = ", ".join([f"{p:.1f} Гц" for p in noise_peaks])
        recommendations.append(f"🔍 Обнаружены выраженные пики вибраций на частотах: {peaks_str}")

        max_peak = max(noise_peaks)

        if max_peak > 300.0:
            recommendations.append(
                "⚙️ **Высокочастотный шум (>300 Гц):** Часто связан с дисбалансом пропеллеров, поврежденными подшипниками моторов "
                "или шумами от регуляторов оборотов (ESC). Рекомендуется проверить механику и сузить полосу динамического фильтра."
            )
            cli_commands.append(f"set dyn_notch_max_hz = {int(min(550, max_peak + 50))}")
            cli_commands.append("set dyn_notch_width_hz = 10")

        elif 150.0 <= max_peak <= 300.0:
            recommendations.append(
                f"⚙️ Резонанс рамы среднего диапазона (~{int(max_peak)} Гц): "
                "Типичная частота для карбоновых рам среднего размера. "
                "Убедитесь, что включен RPM-фильтр. При сильном зуде скорректируйте частоту Gyro Lowpass 2."
            )
            cli_commands.append(f"set gyro_lowpass2_hz = {int(max(130, max_peak - 30))}")

        elif 60.0 <= max_peak < 150.0:
            recommendations.append(
                f"⚠️ Низкочастотный резонанс (~{int(max_peak)} Гц): "
                "Обычно вызван недостаточной жесткостью рамы, люфтами в стэке "
                "или касанием проводов корпуса полетного контроллера. Проверьте механическую сборку перед программным зажатием фильтров."
            )
            cli_commands.append(f"set gyro_lowpass_hz = {int(max(70, max_peak - 15))}")

        recommendations.append(
            "💡 Совет по настройке: "
            "Убедитесь, что задействован двунаправленный RPM-фильтр (Bi-directional DShot). "
            "Это позволяет агрессивнее распускать статические фильтры и сохранять минимальную задержку управления."
        )
        cli_commands.append("save")
    else:
        recommendations.append(
            "✅ Шумовой профиль в норме: "
            "Опасных резонансов выше 50 Гц не обнаружено. "
            "Текущие настройки фильтрации работают оптимально, дополнительное зажатие фильтров не требуется."
        )

    return FilterAnalysisResult(
        noise_peaks=noise_peaks,
        recommendations=recommendations,
        cli_commands=cli_commands
    )
