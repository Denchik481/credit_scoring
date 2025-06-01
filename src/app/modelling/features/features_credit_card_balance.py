import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Пути к кастомным модулям
sys.path.append(r"C:\Users\Denis\credit_scoring\src\app\utils")
sys.path.append(r"C:\Users\Denis\credit_scoring\src\config")

from db_config import DB_ARGS
from db_handler import PostgresDBHandler

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)
np.set_printoptions(suppress=False, precision=2)


def preprocess_credit_card_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка таблицы credit_card_balance:
    - заполнение пропусков
    - приведение типов
    - удаление лишних столбцов
    - минимизация использования памяти
    """
    df = df.copy()

    numeric_fill_zero = [
        "amt_payment_total_current", "amt_drawings_current", "amt_payment_current",
        "amt_balance", "amt_total_receivable", "amt_installment",
        "cnt_instalment_mature_cum", "cnt_drawings_atm_current",
        "cnt_drawings_pos_current"
    ]

    for col in numeric_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["name_contract_status"] = df["name_contract_status"].fillna("Unknown")
    df["sk_dpd"] = df["sk_dpd"].fillna(0)
    df["sk_dpd_def"] = df["sk_dpd_def"].fillna(0)

    df = df.drop(columns=[
        "amt_receivable", "amt_receivable_principal", "amt_drawings_other_current"
    ], errors="ignore")

    # Downcast типов
    for col in df.select_dtypes("int").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("float").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Категориальные
    if "name_contract_status" in df.columns:
        df["name_contract_status"] = df["name_contract_status"].astype("category")

    df.columns = df.columns.str.lower()

    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"[INFO] credit_card_balance: shape={df.shape}, memory={mem:.2f} MB")
    return df


def generate_credit_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков на основе таблицы credit_card_balance.
    Выход: данные на уровне клиента (sk_id_curr).
    """
    df = df.copy()

    # === 1. Базовые расчёты =================================================================================
    df["utilization"] = df["amt_balance"] / df["amt_credit_limit_actual"].replace(0, np.nan)
    df["x"] = df["months_balance"]
    df["x2"] = df["x"] ** 2
    df["x_y_balance"] = df["x"] * df["amt_balance"]
    df["x_y_inst"] = df["x"] * df["amt_inst_min_regularity"]
    df["x_y_util"] = df["x"] * df["utilization"]

    draw_amt_cols = [
        "amt_drawings_current", "amt_drawings_atm_current",
        "amt_drawings_pos_current"
    ]
    draw_cnt_cols = [
        "cnt_drawings_current", "cnt_drawings_atm_current",
        "cnt_drawings_pos_current"
    ]
    df["total_amt_drawings"] = df[draw_amt_cols].sum(axis=1)
    df["total_cnt_drawings"] = df[draw_cnt_cols].sum(axis=1)

    df["payment_balance_ratio"] = df["amt_payment_current"] / df["amt_balance"].replace(0, np.nan)
    df["payment_to_limit_ratio"] = df["amt_payment_current"] / df["amt_credit_limit_actual"].replace(0, np.nan)
    df["payment_to_balance_ratio"] = df["amt_payment_current"] / df["amt_balance"].replace(0, np.nan)

    df["is_late_payment"] = (df["sk_dpd"] > 0).astype(int)
    df["high_util"] = (df["utilization"] > 0.8).astype(int)
    df["has_credit_limit"] = (df["amt_credit_limit_actual"] > 0).astype(int)

    # === 2. Агрегации по временным окнам (все, последние 3/6/12 месяцев) ================================
    def agg_by_period(window=None, suffix=""):
        d = df.copy()
        if window is not None:
            d = d[d["months_balance"] >= -window]
        agg_cols = [
            "amt_balance", "amt_credit_limit_actual", "amt_payment_current",
            "amt_drawings_current"
        ]
        stats = d.groupby("sk_id_curr")[agg_cols].agg(["mean", "sum", "max", "min"])
        stats.columns = [f"{col}_{agg}{suffix}" for col, agg in stats.columns]
        return stats.reset_index()

    agg_all = agg_by_period()
    agg_3 = agg_by_period(3, "_recent_3")
    agg_6 = agg_by_period(6, "_recent_6")
    agg_12 = agg_by_period(12, "_recent_12")

    base = agg_all.merge(agg_3, on="sk_id_curr", how="left") \
                  .merge(agg_6, on="sk_id_curr", how="left") \
                  .merge(agg_12, on="sk_id_curr", how="left")

    for col in [
        "amt_balance_mean", "amt_balance_sum", "amt_credit_limit_actual_mean",
        "amt_payment_current_sum"
    ]:
        for period in ["3", "6", "12"]:
            recent_col = f"{col}_recent_{period}"
            if recent_col in base.columns:
                base[f"{col}_ratio_{period}"] = base[col] / base[recent_col].replace(0, np.nan)

    # === 3. Линейные тренды ========================================================================
    def calc_slope_row(row, y_total, xy_total):
        denom = row["n"] * row["sum_x2"] - row["sum_x"] ** 2
        return (row["n"] * row[xy_total] - row["sum_x"] * row[y_total]) / denom if denom != 0 else np.nan

    trend = df.groupby("sk_id_curr").agg(
        n=("x", "size"),
        sum_x=("x", "sum"), sum_x2=("x2", "sum"),
        sum_y_balance=("amt_balance", "sum"), sum_xy_balance=("x_y_balance", "sum"),
        sum_y_inst=("amt_inst_min_regularity", "sum"), sum_xy_inst=("x_y_inst", "sum"),
        sum_y_util=("utilization", "sum"), sum_xy_util=("x_y_util", "sum")
    ).reset_index()

    trend["balance_trend_slope"] = trend.apply(lambda r: calc_slope_row(r, "sum_y_balance", "sum_xy_balance"), axis=1)
    trend["payment_min_trend"] = trend.apply(lambda r: calc_slope_row(r, "sum_y_inst", "sum_xy_inst"), axis=1)
    trend["utilization_trend_slope"] = trend.apply(lambda r: calc_slope_row(r, "sum_y_util", "sum_xy_util"), axis=1)
    trend = trend[["sk_id_curr", "balance_trend_slope", "payment_min_trend", "utilization_trend_slope"]]

    # === 4. Поведенческие агрегаты по клиенту =========================================================
    behavior = df.groupby("sk_id_curr").agg(
        mean_utilization=("utilization", "mean"),
        late_payment_ratio=("is_late_payment", "mean"),
        max_sk_dpd=("sk_dpd", "max"),
        mean_sk_dpd=("sk_dpd", "mean"),
        total_amt_drawings=("total_amt_drawings", "sum"),
        total_cnt_drawings=("total_cnt_drawings", "sum"),
        total_cnt_installments=("cnt_instalment_mature_cum", "sum"),
        balance_mean=("amt_balance", "mean"),
        credit_limit_mean=("amt_credit_limit_actual", "mean"),
        payment_mean=("amt_payment_current", "mean"),
        payment_balance_ratio=("payment_balance_ratio", "mean"),
        payment_to_limit_mean=("payment_to_limit_ratio", "mean"),
        payment_to_balance_mean=("payment_to_balance_ratio", "mean"),
        high_util_ratio=("high_util", "mean"),
        has_credit_limit_ratio=("has_credit_limit", "mean")
    ).reset_index()

    # === 5. Дельта и % изменение баланса ===============================================================
    sorted_df = df.sort_values("months_balance")
    balance_delta = sorted_df.groupby("sk_id_curr")["amt_balance"].agg(
        lambda x: x.iloc[-1] - x.iloc[0]
    ).reset_index(name="balance_delta")

    pct_change = sorted_df.groupby("sk_id_curr")["amt_balance"].agg(["first", "last"]).reset_index()
    pct_change["balance_pct_change"] = (
        (pct_change["last"] - pct_change["first"]) / pct_change["first"].replace(0, np.nan)
    )
    pct_change = pct_change[["sk_id_curr", "balance_pct_change"]]

    # === 6. Последние значения клиента ==================================================================
    last = df.sort_values("months_balance", ascending=False).groupby("sk_id_curr").first().reset_index()
    last = last[["sk_id_curr", "amt_balance", "amt_credit_limit_actual", "utilization"]]
    last.columns = ["sk_id_curr", "last_amt_balance", "last_credit_limit", "last_utilization"]

    # === 7. Объединение ============================================================================
    features = base \
        .merge(trend, on="sk_id_curr", how="left") \
        .merge(behavior, on="sk_id_curr", how="left") \
        .merge(balance_delta, on="sk_id_curr", how="left") \
        .merge(pct_change, on="sk_id_curr", how="left") \
        .merge(last, on="sk_id_curr", how="left")

    return features


def main():
    """
    Главная функция:
    - загружает данные из базы
    - генерирует признаки по credit_card_balance
    - сохраняет результат в CSV
    """
    start = time.time()

    parser = argparse.ArgumentParser(description="Генерация признаков из credit_card_balance")
    parser.add_argument("--output_file", required=True, help="Путь к итоговому CSV файлу")
    args = parser.parse_args()

    db = PostgresDBHandler(DB_ARGS)
    db.connect()

    df = db.get_df_from_query("SELECT * FROM credit_card_balance")
    df = preprocess_credit_card_balance(df)

    features = generate_credit_card_features(df)
    features.to_csv(args.output_file, index=False)

    print(f"✅ Признаки сохранены: {args.output_file}")
    print(f"⏱ Время выполнения: {time.time() - start:.2f} сек.")


if __name__ == "__main__":
    main()