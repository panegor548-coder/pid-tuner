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
    Анализирует FFT спектр гироскопа по методу независимых субполосных блоков (chunks по 50 Гц).
    """
    if freqs is None or power is None or len(freqs) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: массив частот или мощности пуст."]
        )

    noise_peaks: List[float] = []
    recommendations: List[str] = []
    cli_commands: List[str] = []

    # Отсекаем только самый грязный подвал до 30 Гц
    valid_mask = freqs >= 30.0
    if not np.any(valid_mask):
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: частотная сетка лога слишком узкая (требуется диапазон >30 Гц)."]
        )

    f_valid = freqs[valid_mask]
    p_valid = power[valid_mask]

    if len(p_valid) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Отсутствуют данные гироскопа в диапазоне выше 30 Гц."]
        )

    global_max = np.max(p_valid)
    global_mean = np.mean(p_valid)

    # Проверка на общий чистый лог
    if global_max <= global_mean * 2.0 or global_max < 0.5:
        recommendations.append(
            "⚠️ Лог в рабочей зоне (>30 Гц) выглядит относительно ровным или чистым (нет резких выраженных пиков резонанса)."
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

    # Истинный субполосный метод: проверяем каждый 50 Гц блок автономно
    chunk_size = 50.0
    min_f = 30.0
    max_f = 400.0
    
    candidates = []

    current_start = min_f
    while current_start < max_f:
        current_end = current_start + chunk_size
        
        chunk_mask = (f_valid >= current_start) & (f_valid < current_end)
        if np.any(chunk_mask):
            f_chunk = f_valid[chunk_mask]
            p_chunk = p_valid[chunk_mask]
            
            if len(p_chunk) > 5:
                chunk_max = np.max(p_chunk)
                chunk_median = np.median(p_chunk)
                
                # Если в этом конкретном диапазоне есть пик выше его собственного локального фона
                if chunk_max > chunk_median * 2.5 and chunk_max > 10.0:
                    peak_idx = np.argmax(p_chunk)
                    peak_freq = float(f_chunk[peak_idx])
                    
                    # Добавляем пик, если он не дублирует соседний из этого же чанка
                    if not candidates or all(abs(peak_freq - existing) > 20.0 for existing in candidates):
                        candidates.append(peak_freq)

        current_start = current_end

    # Сортируем найденные локальные пики по частоте (от меньших к большим)
    if candidates:
        noise_peaks = sorted(candidates)

    # Шаг 4: Формирование экспертных рекомендаций на основе найденных частот (методика Криса Россера)
    if noise_peaks:
        peaks_str = ", ".join([f"{p:.1f} Гц" for p in noise_peaks])
        recommendations.append(f"🔍 Обнаружены выраженные пики вибраций по диапазонам: {peaks_str}")

        max_peak = max(noise_peaks)

        if max_peak > 300.0:
            recommendations.append(
                "⚙️ **Высокочастотный шум (>300 Гц):** Часто связан с дисбалансом пропеллеров, поврежденными подшипниками моторов "
                "или шумами от регуляторов оборотов (ESC). Рекомендуется проверить механику и сузить полосу динамического фильтра."
            )
            cli_commands.append(f"set dyn_notch_max_hz = {int(min(550, max_peak + 50))}")
            cli_commands.append(f"set dyn_notch_width_hz = 10")

        elif 150.0 <= max_peak <= 300.0:
            recommendations.append(
                f"⚙️ Резонанс рамы среднего диапазона (~{int(max_peak)} Гц): "
                "Типичная частота для карбоновых рам среднего размера. "
                "Убедитесь, что включен RPM-фильтр. При сильном зуде скорректируйте частоту Gyro Lowpass 2."
            )
            cli_commands.append(f"set gyro_lowpass2_hz = {int(max(130, max_peak - 30))}")

        elif 30.0 <= max_peak < 150.0:
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
            "Опасных резонансов ни в одном из диапазонов не обнаружено. "
            "Текущие настройки фильтрации работают оптимально, дополнительное зажатие фильтров не требуется."
        )

    return FilterAnalysisResult(
        noise_peaks=noise_peaks,
        recommendations=recommendations,
        cli_commands=cli_commands
    )
