import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(r"C:\Users\Denis\credit_scoring\src\app\utils")
sys.path.append(r"C:\Users\Denis\credit_scoring\src\config")
import warnings

from db_config import DB_ARGS
from db_handler import PostgresDBHandler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Отключить все предупреждения
warnings.filterwarnings("ignore")

def preprocess_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка application_train_test перед генерацией признаков.
    Очистка, типизация, заполнение пропусков, снижение памяти.
    """
    df = df.copy()
    
    # === 1. Очистка аномалий =================================================================================

    # Gender — исправим "XNA"
    df["code_gender"] = df["code_gender"].replace("XNA", df["code_gender"].mode()[0])

    # Unknown marital status — возможна проблема
    df = df[df["name_family_status"] != "Unknown"]

    # Occupation — замена NaN
    df["occupation_type"] = df["occupation_type"].fillna("occupation_type_unknown")

    # Аномальный стаж работы (365243 дней)
    df["anomalous_days_employed"] = (df["days_employed"] == 365243).astype("int8")
    df["days_employed"] = df["days_employed"].replace(365243, np.nan)

    # === 2. Заполнение пропусков =============================================================================

    df.fillna({
        "amt_income_total": 1,
        "amt_credit": 1,
        "amt_annuity": 1,
        "amt_goods_price": 1,
        "name_income_type": "Missing",
        "organization_type": "Missing"
    }, inplace=True)

    # === 3. Aggressive Downcasting (типизация) ===============================================================

    # Целочисленные признаки
    int_cols = [
        "cnt_children", "flag_mobil", "flag_emp_phone", "flag_work_phone",
        "flag_phone", "flag_email", "region_rating_client", "region_rating_client_w_city"
    ]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Вещ. признаки (финансовые, возраст, дни)
    float_cols = [
        "amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
        "days_birth", "days_employed", "days_registration", "days_id_publish",
        "ext_source_1", "ext_source_2", "ext_source_3",
        "cnt_fam_members"
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Категориальные признаки
    cat_cols = [
        "code_gender", "name_type_suite", "name_income_type", "name_education_type",
        "name_family_status", "name_housing_type", "occupation_type", "organization_type",
        "weekday_appr_process_start", "fONDKAPREMONT_MODE", "housetype_mode",
        "wallsmaterial_mode", "emergency_state"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # === 4. Флаги и бинарные признаки ========================================================================

    # Флаг отсутствия email + телефона
    df["no_contact_info"] = ((df["flag_phone"] == 0) & (df["flag_email"] == 0)).astype("int8")

    # Флаг наличия детей
    df["has_children"] = (df["cnt_children"] > 0).astype("int8")

    # === 5. Логарифмы для будущих моделей ====================================================================

    for col in ["amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # === 6. Проверка типов и памяти ==========================================================================
    print(f"[INFO] Memory usage after processing: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"[INFO] Data shape: {df.shape}")

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # === 1. Документы ===
    doc_cols = [c for c in df.columns if c.startswith("flag_document_")]
    df["document_sum"] = df[doc_cols].sum(axis=1)
    df.drop(columns=doc_cols, inplace=True)

    # === 2. Информация о жилье ===
    housing_features = [
        "apartments", "basementarea", "years_beginexpluatation", "years_build", "commonarea",
        "elevators", "entrances", "floorsmax", "floorsmin", "landarea", 
        "livingapartments", "livingarea", "nonlivingapartments", "nonlivingarea"
    ]
    housing_stats = ["avg", "mode", "medi"]
    housing_cols = [
        col for col in df.columns
        if any(col.startswith(f) and col.endswith(stat) for f in housing_features for stat in housing_stats)
        or col in {"fondkapremont_mode", "housetype_mode", "wallsmaterial_mode", "emergencystate_mode", "totalarea_mode"}
    ]
    df["flag_full_info_house"] = (df[housing_cols].notna().sum(axis=1) < 30).astype(int)
    df.drop(columns=housing_cols, inplace=True)

    # === 3. Идентификационные данные ===
    df["year_birth"] = (-df["days_birth"] / 365.25).astype(int)
    df["years_since_id_change"] = (-df["days_id_publish"] / 365.25).astype(int)
    df["id_change_years"] = df["year_birth"] - df["years_since_id_change"]
    df["document_change_delay"] = (~df["id_change_years"].isin([14, 20, 45])).astype(int)

    # === 4. Финансовые соотношения ===
    df["income_annuity_ratio"] = df["amt_annuity"] * 12 / df["amt_income_total"].replace(0, np.nan)
    df["credit_vs_income_ratio"] = df["amt_credit"] / df["amt_income_total"].replace(0, np.nan)
    df["annuity_credit_ratio"] = df["amt_annuity"] / df["amt_credit"].replace(0, np.nan)
    df["goods_credit_ratio"] = df["amt_goods_price"] / df["amt_credit"].replace(0, np.nan)
    df["credit_minus_goods"] = df["amt_credit"] - df["amt_goods_price"]
    df["credit_annuity_goods"] = df["amt_annuity"] / (df["amt_credit"] + df["amt_goods_price"]).replace(0, np.nan)
    df["disposable_income"] = df["amt_income_total"] - df["amt_annuity"]
    df["monthly_payment_pct"] = df["amt_annuity"] / (df["amt_income_total"] / 12).replace(0, np.nan)
    df["monthly_disposable"] = df["amt_income_total"] - df["amt_annuity"]

    # === 5. Семья и дети ===
    df["avg_count_children_per_adult"] = df["cnt_children"] / df["cnt_fam_members"].replace(0, np.nan)
    df["avg_income_per_child"] = df["amt_income_total"] / df["cnt_children"].replace(0, np.nan)
    df["avg_income_per_adult"] = df["amt_income_total"] / (df["cnt_fam_members"] - df["cnt_children"]).replace(0, np.nan)
    df["debt_per_family_member"] = df["amt_credit"] / (df["cnt_fam_members"] + 1)

    # === 6. География и стабильность адреса ===
    df["region_mismatch_count"] = df[[
        "reg_region_not_live_region", "reg_region_not_work_region", "live_region_not_work_region"
    ]].sum(axis=1)
    df["city_mismatch_count"] = df[[
        "reg_city_not_live_city", "reg_city_not_work_city", "live_city_not_work_city"
    ]].sum(axis=1)
    df["is_fully_stable"] = ((df["region_mismatch_count"] == 0) & (df["city_mismatch_count"] == 0)).astype(int)
    df["fully_mismatched"] = (
        (df["reg_region_not_live_region"] == 1) &
        (df["reg_region_not_work_region"] == 1) &
        (df["live_region_not_work_region"] == 1)
    ).astype(int)
    df["region_rating_gap"] = df["region_rating_client"] - df["region_rating_client_w_city"]
    df["pop_rating_product"] = df["region_population_relative"] * df["region_rating_client"]

    region_mismatch_cols = [
        "region_mismatch_count", "city_mismatch_count", "live_city_not_work_city",
        "reg_city_not_work_city", "live_region_not_work_region", "reg_region_not_work_region"
    ]
    df["region_mismatch_total"] = df[region_mismatch_cols].sum(axis=1)

    # === 7. Работа и стаж ===
    df["employment_age_ratio"] = df["days_employed"] / df["days_birth"].replace(0, np.nan)
    df["last_job_duration_over_age"] = df["days_employed"].abs() / df["days_birth"].abs().replace(0, np.nan)
    df["days_since_employment_change"] = df["days_employed"] - df["days_id_publish"]
    df["days_since_reg"] = df["days_id_publish"] - df["days_registration"]
    df["recent_activity_indicator"] = df["days_last_phone_change"] - df["days_id_publish"]

    # === 8. Социальное окружение ===
    df["def_obs_ratio_30"] = df["def_30_cnt_social_circle"] / (df["obs_30_cnt_social_circle"] + 1)
    df["def_obs_ratio_60"] = df["def_60_cnt_social_circle"] / (df["obs_60_cnt_social_circle"] + 1)
    df["has_defaulters_around"] = ((df["def_30_cnt_social_circle"] + df["def_60_cnt_social_circle"]) > 0).astype(int)

    # === 9. Бинарные признаки и категории риска ===
    df["is_single_parent"] = ((df["cnt_children"] > 0) & (df["cnt_fam_members"] <= 2)).astype(int)
    df["is_big_credit_low_income"] = ((df["amt_credit"] > 1_000_000) & (df["amt_income_total"] < 100_000)).astype(int)
    df["young_age"] = (df["year_birth"] > 1990).astype(int)
    df["old_age"] = (df["year_birth"] < 1960).astype(int)
    df["children_presence"] = (df["cnt_children"] > 0).astype(int)
    df["owns_realty_and_car"] = ((df["flag_own_car"] == 1) & (df["flag_own_realty"] == 1)).astype(int)
    df["owns_neither"] = ((df["flag_own_car"] == 0) & (df["flag_own_realty"] == 0)).astype(int)
    df["is_employed_and_educated"] = (
        (df["name_income_type"] == "Working") & 
        (df["name_education_type"].isin(["Higher education", "Academic degree"]))
    ).astype(int)
    df["is_low_educated"] = df["name_education_type"].isin(["Lower secondary", "Secondary / secondary special"]).astype(int)
    df["is_high_risk_occupation"] = df["occupation_type"].isin(["Laborers", "Drivers", "Security staff"]).astype(int)
    df["long_stable_job"] = ((df["days_employed"] < -4000) & (df["occupation_type"].notna())).astype(int)
    df["short_employment_gap"] = (df["days_employed"] > -1000).astype(int)

    # === 10. Контактные данные и активность ===
    df["has_mobile_and_email"] = ((df["flag_mobil"] == 1) & (df["flag_email"] == 1)).astype(int)
    df["num_active_phones"] = df[["flag_mobil", "flag_phone", "flag_work_phone", "flag_emp_phone", "flag_cont_mobile"]].sum(axis=1)
    df["contactability_score"] = df[["flag_mobil", "flag_phone", "flag_work_phone", "flag_email"]].sum(axis=1)

    # === 11. Кредитная активность в бюро ===
    req_cols = [
        "amt_req_credit_bureau_hour", "amt_req_credit_bureau_day", "amt_req_credit_bureau_week",
        "amt_req_credit_bureau_mon", "amt_req_credit_bureau_qrt", "amt_req_credit_bureau_year"
    ]
    df["request_intensity"] = df[req_cols].sum(axis=1)
    df["recent_bureau_activity"] = ((df["amt_req_credit_bureau_day"] > 0) | (df["amt_req_credit_bureau_week"] > 0)).astype(int)

    # === 12. Дополнительные агрегаты и логарифмы ===
    df["income_per_fam_member"] = df["amt_income_total"] / df["cnt_fam_members"].replace(0, np.nan)
    df["credit_annuity_ratio_log"] = np.log1p(df["credit_annuity_goods"].replace(0, np.nan))
    df["annuity_income_ratio_log"] = np.log1p(df["income_annuity_ratio"].replace(0, np.nan))
    df["credit_income_ratio_log"] = np.log1p(df["credit_vs_income_ratio"].replace(0, np.nan))
    df["credit_income_interaction"] = df["amt_credit"] * df["amt_income_total"]
    df["doc_known_score"] = df["document_sum"] + (1 - df["flag_full_info_house"])
    df["has_weak_docs"] = (df["doc_known_score"] < 8).astype(int)
    df["is_semi_stable"] = (
        (df["region_mismatch_count"] + df["city_mismatch_count"] >= 1) &
        (df["is_fully_stable"] == 0)
    ).astype(int)
    df["total_social_obs"] = df[["obs_30_cnt_social_circle", "obs_60_cnt_social_circle"]].sum(axis=1)
    df["total_social_def"] = df[["def_30_cnt_social_circle", "def_obs_ratio_60"]].sum(axis=1)

    return df

def compute_weighted_score(df: pd.DataFrame) -> pd.Series:
    ext_sources = ["ext_source_1", "ext_source_2", "ext_source_3"]
    weights = 1 - df[ext_sources].isna().mean()
    return (df[ext_sources] * weights.values).sum(axis=1)


def train_and_predict_rate(application_df, db_handler):
    query = """
    SELECT 
        rate_interest_primary,
        amt_annuity, amt_credit, amt_goods_price,
        name_contract_type
    FROM previous_application
    """
    df_prev = db_handler.get_df_from_query(query).dropna()
    X = df_prev[["amt_annuity", "amt_credit", "amt_goods_price", "name_contract_type"]]
    y = df_prev["rate_interest_primary"]

    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
             ["amt_annuity", "amt_credit", "amt_goods_price"]),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]),
             ["name_contract_type"])
        ])),
        ("model", GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_test_pred = pipeline.predict(X_test)
    print(f"RMSE: {mean_squared_error(y_test, y_test_pred, squared=False):.4f}")
    print(f"R2: {r2_score(y_test, y_test_pred):.4f}")

    X_app = application_df[X.columns]
    return pipeline.predict(X_app)

def main():
    start = time.time()

    # Аргументы
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    # Подключение к БД
    db = PostgresDBHandler(DB_ARGS)
    db.connect()

    # Загрузка
    df = db.get_df_from_query("SELECT * FROM application_train_test")
    df = preprocess_application(df)

    # Предикт рейтинга
    df["predicted_rate_interest_primary"] = train_and_predict_rate(df, db)

    # Взв. внешний скор
    df["weighted_score"] = compute_weighted_score(df)

    # Групповой средний доход
    train = df[df["target"].notna()].copy()
    test = df[df["target"].isna()].copy()
    group_income = train.groupby(["code_gender", "name_education_type"])["amt_income_total"].mean()

    train["group_mean_income"] = train.set_index(["code_gender", "name_education_type"]).index.map(group_income)
    test["group_mean_income"] = test.set_index(["code_gender", "name_education_type"]).index.map(group_income)

    df = pd.concat([train, test]).sort_index()

    # Основные признаки
    df = generate_features(df)

    # Сохраняем
    df.to_csv(args.output_file, index=False)
    print(f"Сохранено: {args.output_file}")
    print(f"Время выполнения: {time.time() - start:.2f} сек.")

if __name__ == "__main__":
    main() 
