"""Модуль для определения результатов скоринга и фичей для принятия решений."""

from dataclasses import dataclass
from enum import Enum, auto


class ScoringDecision(Enum):
    """Возможные решения модели."""

    ACCEPTED = auto()
    DECLINED = auto()


@dataclass
class ScoringResult:
    """Класс, содержащий результаты скоринга."""

    decision: ScoringDecision
    amount: int
    threshold: float
    proba: float


@dataclass
class Features:
    """Фичи для принятия решения об одобрении."""

    document_sum: float
    year_birth: float
    weighted_score: float
    credit_vs_income_ratio: float
    predicted_rate_interest_primary: float
    annuity_credit_ratio: float
    pop_rating_product: float
    last_job_duration_over_age: float
    days_last_phone_change: float
    days_registration: float
    goods_credit_ratio: float
    name_contract_type: str = "other"  # Аргумент с значением по умолчанию перемещён в конец

class AntiFraudRules:
    @staticmethod
    def is_fraud(features: Features) -> bool:
        flag_recent_phone_change = (features.days_last_phone_change > -10)
        flag_recent_registration = (features.days_registration > -10)
        flag_last_job = (features.last_job_duration_over_age < 0.01)
        flag_goods_credit_ratio = (features.goods_credit_ratio > 2)

        return (
            (flag_recent_phone_change and flag_recent_registration)
            or flag_last_job
            or flag_goods_credit_ratio
        )