"""Модуль для расчёта одобренной суммы кредита."""
import sys
sys.path.append(r"c:\Users\Denis\credit_scoring\src")

from app.core.api import Features


class Calculator:
    """Класс для расчёта одобренной суммы кредита на основе вероятности дефолта и признаков клиента."""

    def calc_amount(self, proba: float, features: Features) -> int:
        """
        Рассчитывает одобренную сумму кредита.

        :param proba: Вероятность дефолта.
        :param features: Признаки клиента.
        :return: Одобренная сумма кредита.
        """
        # Минимальный риск — максимальная сумма
        if (
            proba < 0.02
            and features.credit_vs_income_ratio < 3.5
            and features.year_birth > 45
        ):
            return 500_000
        # Низкий риск
        if (
            proba < 0.05
            and features.credit_vs_income_ratio < 4
            and features.year_birth > 40
        ):
            return 400_000
        # Средний риск
        if (
            proba < 0.1
            and features.credit_vs_income_ratio < 4.5
            and features.year_birth > 36
        ):
            return 300_000
        # Повышенный риск
        if proba < 0.15:
            return 200_000
        # Высокий риск — минимальная сумма
        return 100_000
