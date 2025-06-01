import pandas as pd
from catboost import CatBoostClassifier
from core.api import Features, ScoringDecision, ScoringResult, AntiFraudRules
from core.calculator import Calculator


class SimpleModel(object):
    """Класс для моделей c расчетом proba и threshold."""

    _threshold = 0.3
    _approved_amount = 100_000

    def __init__(self, model_path: str):
        """Создает объект класса."""
        self._model = CatBoostClassifier()
        self._model.load_model(model_path)
    
    

    def get_scoring_result(self, features: Features) -> ScoringResult:
        if AntiFraudRules.is_fraud(features):
            return ScoringResult(
                decision=ScoringDecision.DECLINED,
                amount=0,
                threshold=self._threshold,
                proba=1.0,
            )

        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0
        if proba < self._threshold:
            amount = self._approved_amount
            decision = ScoringDecision.ACCEPTED

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )

    def _predict_proba(self, features: Features) -> float:
        data = pd.DataFrame(
            [
                {
                    "document_sum": features.document_sum,
                    "year_birth": features.year_birth,
                    "weighted_score": features.weighted_score,
                    "credit_vs_income_ratio": features.credit_vs_income_ratio,
                    "predicted_rate_interest_primary": features.predicted_rate_interest_primary,
                    "annuity_credit_ratio": features.annuity_credit_ratio,
                    "pop_rating_product": features.pop_rating_product,
                    "last_job_duration_over_age": features.last_job_duration_over_age,
                    "name_contract_type": features.name_contract_type,
                    "days_last_phone_change": features.days_last_phone_change,
                    "days_registration": features.days_registration,
                    "goods_credit_ratio": features.goods_credit_ratio,
                }
            ]
        )
        return self._model.predict_proba(data)[0, 1]


class AdvancedModel(SimpleModel):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._calculator = Calculator()

    def get_scoring_result(self, features: Features) -> ScoringResult:
        if AntiFraudRules.is_fraud(features):
            return ScoringResult(
                decision=ScoringDecision.DECLINED,
                amount=0,
                threshold=self._threshold,
                proba=1.0,
            )

        proba = self._predict_proba(features)

        decision = ScoringDecision.DECLINED
        amount = 0
        if proba < 0.3:
            decision = ScoringDecision.ACCEPTED
            amount = self._calculator.calc_amount(
                proba,
                features,
            )

        return ScoringResult(
            decision=decision,
            amount=amount,
            threshold=self._threshold,
            proba=proba,
        )
