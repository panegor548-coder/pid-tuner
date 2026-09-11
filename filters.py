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
    Анализирует FFT спектр гироскопа, ищет пики резонансов и формирует рекомендации для фильтров.
    """
    if freqs is None or power is None or len(freqs) == 0:
        return FilterAnalysisResult(
            recommendations=["Недостаточно данных для анализа частот."]
        )

    noise_peaks = []
    recommendations = []
    cli_commands = []

    valid_mask = freqs > 10  # отсекаем движения стиками
    if np.any(valid_mask):
        f_valid = freqs[valid_mask]
        p_valid = power[valid_mask]
        
        mean_power = np.mean(p_valid)
        peak_mask = (p_valid > mean_power * 5.0)
        if np.any(peak_mask):
            peak_freqs = f_valid[peak_mask]
            for f in peak_freqs:
                if not noise_peaks or all(abs(f - p) > 15 for p in noise_peaks):
                    noise_peaks.append(float(f))
            noise_peaks = sorted(noise_peaks[:3])

    if noise_peaks:
        recommendations.append(f"Обнаружены пики вибраций на частотах: {', '.join([f'{p:.1f} Гц' for p in noise_peaks])}")
        max_peak = max(noise_peaks)
        
        if max_peak > 250:
            recommendations.append("Высокочастотный шум. Рекомендация: проверить пропеллеры и поднять Dynamic Notch Hz.")
            cli_commands.append(f"set dyn_notch_max_hz = {int(min(600, max_peak + 50))}")
        elif 80 <= max_peak <= 250:
            recommendations.append(f"Резонанс рамы в районе ~{int(max_peak)} Гц. Рекомендация: настроить Gyro Lowpass.")
            cli_commands.append(f"set gyro_lowpass_hz = {int(max(80, max_peak - 20))}")
        
        cli_commands.append("save")
    else:
        recommendations.append("Шумовой профиль в норме, выраженных резонансов не обнаружено.")

    return FilterAnalysisResult(
        noise_peaks=noise_peaks,
        recommendations=recommendations,
        cli_commands=cli_commands
    )