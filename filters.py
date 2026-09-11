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
    Анализирует FFT спектр гироскопа по методу независимых субполосных блоков (chunks по 50 Гц)
    с адаптивным поиском пиков и обязательным предупреждением для визуальной проверки пользователем.
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

    # Субполосный метод: независимая проверка каждого 50 Гц блока по локальной медиане
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
                
                # Адаптивный поиск: пик должен заметно выделяться на фоне своего локального окружения (в 4 раза)
                if chunk_max > chunk_median * 4.0:
                    peak_idx = np.argmax(p_chunk)
                    peak_freq = float(f_chunk[peak_idx])
                    
                    if not candidates or all(abs(peak_freq - existing) > 20.0 for existing in candidates):
                        candidates.append(peak_freq)

        current_start = current_end

    if candidates:
        noise_peaks = sorted(candidates)

    # Формирование экспертных рекомендаций (методика Криса Россера)
    if noise_peaks:
        peaks_str = ", ".join([f"{p:.1f} Гц" for p in noise_peaks])
        recommendations.append(f"🔍 Автоматика обнаружила пики на частотах: {peaks_str}")
        
        # ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ ДЛЯ ПОЛЬЗОВАТЕЛЯ
        recommendations.append(
            "⚠️ **Внимание — визуальная проверка:** Обязательно посмотрите на график спектра глазами! "
            "Если линия на указанных частотах выглядит абсолютно ровной и «плоской» без резких торчащих бугров, "
            "это ложное срабатывание алгоритма на фоне шума. **Не применяйте CLI-команды**, если график чистый."
        )

        max_peak = max(noise_peaks)

        if max_peak > 300.0:
            recommendations.append(
                "⚙️ **Высокочастотный шум (>300 Гц):** Возможен дисбаланс пропеллеров или проблемы с подшипниками моторов."
            )
            cli_commands.append(f"set dyn_notch_max_hz = {int(min(550, max_peak + 50))}")
            cli_commands.append("set dyn_notch_width_hz = 10")

        elif 150.0 <= max_peak <= 300.0:
            recommendations.append(
                f"⚙️ Резонанс рамы (~{int(max_peak)} Гц): типичный диапазон для средних рам. Убедитесь, что включен RPM-фильтр."
            )
            cli_commands.append(f"set gyro_lowpass2_hz = {int(max(130, max_peak - 30))}")

        elif 30.0 <= max_peak < 150.0:
            recommendations.append(
                f"⚠️ Низкочастотный резонанс (~{int(max_peak)} Гц): проверьте жесткость рамы, стэк и укладку провода USB/питания."
            )
            cli_commands.append(f"set gyro_lowpass_hz = {int(max(70, max_peak - 15))}")

        recommendations.append(
            "💡 Базовое правило: если сомневаетесь в наличии пика на графике — лучше оставьте фильтры дефолтными."
        )
        cli_commands.append("save")
    else:
        recommendations.append(
            "✅ Шумовой профиль в норме: опасных резонансов ни в одном из диапазонов не обнаружено. Фильтры менять не нужно."
        )

    return FilterAnalysisResult(
        noise_peaks=noise_peaks,
        recommendations=recommendations,
        cli_commands=cli_commands
    )
