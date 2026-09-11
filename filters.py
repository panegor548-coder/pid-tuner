"""
filters.py — Модуль анализа спектра шумов и формирования рекомендаций по фильтрам Betaflight.
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
    Если данные отсутствуют или сигнал плоский, возвращает точную причину.
    """
    if freqs is None or power is None or len(freqs) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: массив частот или мощности пуст."]
        )

    noise_peaks = []
    recommendations = []
    cli_commands = []

    # Проверяем диапазон частот выше 80 Гц (где живут резонансы)
    valid_mask = freqs > 80
    if not np.any(valid_mask):
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: частотная сетка лога слишком узкая для поиска резонансов (>80 Гц)."]
        )

    f_valid = freqs[valid_mask]
    p_valid = power[valid_mask]
    
    if len(p_valid) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Отсутствуют данные гироскопа в диапазоне выше 80 Гц."]
        )

    max_p = np.max(p_valid)
    mean_p = np.mean(p_valid)

    # Строгая проверка на «плоский» или пустой лог
    if max_p < 10.0 or max_p <= mean_p * 4.0:
        return FilterAnalysisResult(
            recommendations=[
                "⚠️ Лог выглядит плоским или отфильтрованным (нет выраженных пиков шума >80 Гц).",
                "💡 Что проверить: убедитесь, что в Blackbox включен лог сырого гироскопа (pre-filter) и в логе есть реальные полеты, а не статичные данные."
            ]
        )

    # Если данные есть, ищем реальные пики
    peak_mask = (p_valid > mean_p * 5.0) & (p_valid > 10.0)
    if np.any(peak_mask):
        peak_freqs = f_valid[peak_mask]
        for f in peak_freqs:
            if not noise_peaks or all(abs(f - p) > 15 for p in noise_peaks):
                noise_peaks.append(float(f))
        noise_peaks = sorted(noise_peaks[:3])

    if noise_peaks:
        recommendations.append(f"🔍 Обнаружены пики вибраций на частотах: {', '.join([f'{p:.1f} Гц' for p in noise_peaks])}")
        max_peak = max(noise_peaks)
        
        if max_peak > 250:
            recommendations.append("⚙️ Высокочастотный шум. Рекомендация: проверить пропеллеры и поднять Dynamic Notch Hz.")
            cli_commands.append(f"set dyn_notch_max_hz = {int(min(600, max_peak + 50))}")
        elif 80 <= max_peak <= 250:
            recommendations.append(f"⚙️ Резонанс рамы в районе ~{int(max_peak)} Гц. Рекомендация: настроить Gyro Lowpass.")
            cli_commands.append(f"set gyro_lowpass_hz = {int(max(80, max_peak - 20))}")
        
        cli_commands.append("save")
    else:
        recommendations.append("✅ Шумовой профиль в норме, выраженных резонансов в диапазоне >80 Гц не обнаружено.")

    return FilterAnalysisResult(
        noise_peaks=noise_peaks,
        recommendations=recommendations,
        cli_commands=cli_commands
    )
