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
    Если данные отсутствуют или сигнал плоский, возвращает точную причину.
    При обнаружении пиков формирует рекомендации с учетом задержки и рекомендаций Криса Россера.
    """
    if freqs is None or power is None or len(freqs) == 0:
        return FilterAnalysisResult(
            recommendations=["❌ Недостаточно данных: массив частот или мощности пуст."]
        )

    noise_peaks = []
    recommendations = []
    cli_commands = []

    # Проверяем диапазон частот выше 80 Гц (где живут резонансы рамы и моторов)
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

    # Если данные есть, ищем реальные пики (учитываем методологию Россера: отсечение ложных гармоник)
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
        
        # Эвристика по гайду Криса Россера:
        # - Низкие пики (<150 Гц) обычно связаны с жесткостью рамы или затянутыми винтами моторов.
        # - Средние и высокие пики (>200-250 Гц) — работа пропеллеров, подшипники или шумы ESC.
        if max_peak > 300:
            recommendations.append(
                "⚙️ Обнаружен мощный высокочастотный шум (>300 Гц). "
                "Рекомендуется проверить состояние подшипников моторов, балансировку пропеллеров и сузить Dynamic Notch Hz."
            )
            cli_commands.append(f"set dyn_notch_max_hz = {int(min(550, max_peak + 50))}")
            cli_commands.append("set dyn_notch_width_hz = 10")
        elif 150 <= max_peak <= 300:
            recommendations.append(
                f"⚙️ Выявлен резонанс рамы в диапазоне ~{int(max_peak)} Гц. "
                "По методологии Betaflight 4.4+, если RPM-фильтр включен, убедитесь, что добротность (Q) фильтров не глушит полезный сигнал. "
                "При сильном зуде рекомендуется настроить Gyro Lowpass 2."
            )
            cli_commands.append(f"set gyro_lowpass2_hz = {int(max(150, max_peak - 30))}")
        elif 80 <= max_peak < 150:
            recommendations.append(
                f"⚠️ Низкочастотный резонанс рамы (~{int(max_peak)} Гц). "
                "Часто вызван люфтами в раме, незакрепленным стеком или дефектами карбона. Проверьте сборку перед программным зажатием фильтров."
            )
            cli_commands.append(f"set gyro_lowpass_hz = {int(max(90, max_peak - 20))}")

        # Общая рекомендация по RPM-фильтрации (советы сообщества)
        recommendations.append("💡 Убедитесь, что задействован RPM-фильтр (Dshot), так как он позволяет агрессивнее распускать статические фильтры и снижать задержку управления.")
        cli_commands.append("save")
    else:
        recommendations.append(
            "✅ Шумовой профиль чистый, выраженных резонансов в диапазоне >80 Гц не обнаружено. "
            "Фильтрация работает оптимально, дополнительное зажатие фильтров не требуется (сохраняется минимальная задержка управления)."
        )

    return FilterAnalysisResult(
        noise_peaks=noise_peaks,
        recommendations=recommendations,
        cli_commands=cli_commands
    )
