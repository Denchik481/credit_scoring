"""Модуль для генерации признаков из таблицы previous_application."""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from db_config import DB_ARGS
from db_handler import PostgresDBHandler

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
np.set_printoptions(suppress=False, precision=2)
pd.set_option("display.float_format", "{:.2f}".format)


def preprocess_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка таблицы previous_application:
    - очистка пропусков
    - downcast чисел
    - типизация категориальных переменных
    - стандартизация имён столбцов
    - удаление неинформативных значений

    Возвращает оптимизированный DataFrame.
    """
    df = df.copy()

    # === 1. Стандартизируем имена колонок
    df.columns = df.columns.str.lower()

    # === 2. Заполняем пропуски ==========================================================

    # Категориальные — 'XNA' или 'Missing'
    cat_na_fill = {
        "name_contract_type": "Missing",
        "name_cash_loan_purpose": "Missing",
        "name_payment_type": "Missing",
        "product_combination": "Missing",
        "name_client_type": "Missing",
        "name_portfolio": "Missing",
        "name_contract_status": "Missing",
        "code_reject_reason": "NoReason",
    }

    for col, value in cat_na_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # Числовые — 0 или медиана (по ситуации)
    numeric_na_fill = {
        "amt_down_payment": 0,
        "rate_down_payment": 0,
        "amt_credit": 0,
        "amt_application": 1,
        "amt_goods_price": 0,
        "nflag_insured_on_approval": 0,
    }

    for col, value in numeric_na_fill.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    # === 3. Downcast типов ==============================================================

    for col in df.select_dtypes(include="float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include="int").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # === 4. Преобразуем категориальные текстовые признаки ================================

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # === 5. Feature-ready flags ==========================================================

    # Флаг одобрения заявки
    if "name_contract_status" in df.columns:
        df["was_approved"] = (df["name_contract_status"] == "Approved").astype("int8")

    # Флаг отказа
    if "name_contract_status" in df.columns:
        df["was_refused"] = (df["name_contract_status"] == "Refused").astype("int8")

    # Разность между amt_credit и amt_application
    if "amt_credit" in df.columns and "amt_application" in df.columns:
        df["application_diff_to_credit"] = df["amt_credit"] - df["amt_application"]

    # Разность между application и goods_price
    if "amt_goods_price" in df.columns and "amt_application" in df.columns:
        df["application_diff_to_goods"] = df["amt_application"] - df["amt_goods_price"]

    # Плановая длительность кредита
    if "days_last_due" in df.columns and "days_decision" in df.columns:
        df["planned_duration"] = df["days_last_due"] - df["days_decision"]

    # === 6. Очистка сильно редких признаков / бесполезных
    drop_cols = ["flag_last_appl_per_contract", "nflag_last_appl_in_day"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # === 7. Вывод инфо
    mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"[INFO] previous_application: shape={df.shape}, memory={mem:.2f} MB")

    return df


def generate_previous_application_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Продвинутая генерация признаков из previous_application по sk_id_curr.
    """

    df = df.copy()
    
    # Бинарные флаги и расчёты
    df["was_approved"] = (df["name_contract_status"] == "Approved").astype(int)
    df["was_refused"] = (df["name_contract_status"] == "Refused").astype(int)
    df["credit_to_application_ratio"] = df["amt_credit"] / df["amt_application"].replace(0, np.nan)
    df["goods_to_credit_ratio"] = df["amt_goods_price"] / df["amt_credit"].replace(0, np.nan)
    df["downpayment_to_application"] = df["rate_down_payment"] / df["amt_application"].replace(0, np.nan)
    df["has_cash_loan_unknown_purpose"] = (df["name_cash_loan_purpose"] == "XNA").astype(int)
    df["is_cash"] = (df["name_contract_type"] == "Consumer loans").astype(int)
    df["application_diff_to_credit"] = df["amt_credit"] - df["amt_application"]
    df["application_diff_to_goods"] = df["amt_application"] - df["amt_goods_price"]
    df["planned_duration"] = df["days_last_due"] - df["days_decision"]

    agg = df.groupby("sk_id_curr").agg(
        prev_app_count=("sk_id_prev", "count"),
        prev_app_approved_ratio=("was_approved", "mean"),
        prev_app_refused_ratio=("was_refused", "mean"),
        prev_app_cash_loan_unknown_share=("has_cash_loan_unknown_purpose", "mean"),
        prev_app_cash_share=("is_cash", "mean"),

        # AMT CREDIT
        prev_amt_credit_sum=("amt_credit", "sum"),
        prev_amt_credit_mean=("amt_credit", "mean"),
        prev_amt_credit_max=("amt_credit", "max"),
        prev_amt_credit_std=("amt_credit", "std"),

        # AMT APPLICATION
        prev_amt_application_mean=("amt_application", "mean"),
        prev_credit_to_app_ratio_mean=("credit_to_application_ratio", "mean"),
        prev_amt_app_minus_credit_mean=("application_diff_to_credit", "mean"),
        
        # GOODS
        prev_amt_goods_price_mean=("amt_goods_price", "mean"),
        prev_goods_to_credit_ratio_mean=("goods_to_credit_ratio", "mean"),
        prev_amt_app_minus_goods_mean=("application_diff_to_goods", "mean"),

        # Downpayment
        prev_down_payment_rate_mean=("rate_down_payment", "mean"),
        prev_downpayment_to_app_mean=("downpayment_to_application", "mean"),

        # Даты
        prev_days_decision_min=("days_decision", "min"),
        prev_days_decision_max=("days_decision", "max"),
        prev_days_decision_mean=("days_decision", "mean"),
        prev_days_decision_std=("days_decision", "std"),

        # Planned duration
        prev_planned_duration_mean=("planned_duration", "mean"),
        prev_planned_duration_std=("planned_duration", "std"),

        # Дополнительные фичи
        prev_most_common_status=("name_contract_status", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        prev_most_common_cash_purpose=("name_cash_loan_purpose", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        prev_most_common_weekday=("weekday_appr_process_start", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
        prev_unique_contracts=("name_contract_type", "nunique"),
        prev_unique_purposes=("name_cash_loan_purpose", "nunique"),

        # Payment flags
        prev_credit_more_than_app_count=("application_diff_to_credit", lambda x: (x < 0).sum()),
        prev_app_below_goods_count=("application_diff_to_goods", lambda x: (x < 0).sum())
    ).reset_index()

    # Производные признаки
    agg["prev_application_time_span"] = agg["prev_days_decision_max"] - agg["prev_days_decision_min"]
    agg["prev_credit_per_application"] = agg["prev_amt_credit_sum"] / agg["prev_app_count"].replace(0, np.nan)
    agg["prev_rejected_and_approved_mix_flag"] = ((agg["prev_app_approved_ratio"] > 0) & (agg["prev_app_refused_ratio"] > 0)).astype(int)
    agg["prev_credit_to_goods_diff"] = agg["prev_amt_credit_mean"] - agg["prev_amt_goods_price_mean"]

    return agg


def main():
    start_time = time.time()

    # ==== Аргументы командной строки ====
    parser = argparse.ArgumentParser(description="Генерация признаков из previous_application.")
    parser.add_argument(
        "--output_file",
        required=True,
        help="Путь для сохранения итогового CSV-файла с признаками.",
    )
    args = parser.parse_args()

    # ==== Подключение к БД ====
    db_handler = PostgresDBHandler(db_args=DB_ARGS)
    db_handler.connect()

    # ==== SQL-запрос к previous_application ====
    query = """
        SELECT *
        FROM previous_application
    """
    df = db_handler.get_df_from_query(query)
    df = preprocess_previous_application(df)
    # ==== Генерация признаков ====
    features = generate_previous_application_features_v2(df)

    # ==== Сохранение результатов ====
    features.to_csv(args.output_file, index=False)
    print(f"✅ Файл с признаками сохранен по пути: {args.output_file}")
    print(f"⏱ Время выполнения скрипта: {time.time() - start_time:.2f} секунд")


if __name__ == "__main__":
    main()