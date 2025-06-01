import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Пути к модулям проекта
sys.path.append(r"C:\Users\Denis\credit_scoring\src\app\utils")
sys.path.append(r"C:\Users\Denis\credit_scoring\src\config")

from db_config import DB_ARGS
from db_handler import PostgresDBHandler

# Отключение предупреждений
warnings.filterwarnings("ignore")


def preprocess_bureau_and_balance(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Предобработка таблиц bureau и bureau_balance:
    - downcast числовых значений
    - обработка пропусков
    - типизация категорий
    """
    bureau = bureau.copy()
    bureau["days_enddate_fact"] = bureau["days_enddate_fact"].fillna(0)

    # Downcast числовых колонок
    for col in ["days_credit", "days_enddate_fact"]:
        if col in bureau.columns:
            bureau[col] = pd.to_numeric(bureau[col], downcast="integer")

    if "credit_active" in bureau.columns:
        bureau["credit_active"] = bureau["credit_active"].astype("category")

    # Инициализация bureau_balance
    bureau_balance = bureau_balance.copy()
    bureau_balance["status"] = bureau_balance["status"].astype("category")
    bureau_balance["months_balance"] = pd.to_numeric(
        bureau_balance["months_balance"], downcast="integer"
    )

    print(f"[bureau]           shape={bureau.shape}, "
          f"memory={bureau.memory_usage(deep=True).sum() / 1024**2:.2f}MB")
    print(f"[bureau_balance]   shape={bureau_balance.shape}, "
          f"memory={bureau_balance.memory_usage(deep=True).sum() / 1024**2:.2f}MB")

    return bureau, bureau_balance


def load_data(db: PostgresDBHandler) -> pd.DataFrame:
    """
    Загружает и объединяет таблицы bureau и bureau_balance.
    """
    bureau = db.get_df_from_query("""
        SELECT 
            sk_id_curr, sk_bureau_id, credit_active, 
            days_enddate_fact, days_credit, amt_credit_sum, 
            amt_annuity
        FROM bureau
    """)
    bureau_balance = db.get_df_from_query("SELECT * FROM bureau_balance")

    bureau, bureau_balance = preprocess_bureau_and_balance(bureau, bureau_balance)
    df = bureau_balance.merge(bureau, on="sk_bureau_id", how="left")
    return df


def generate_bb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков по таблицам bureau + bureau_balance:
    - статистики по просрочкам, статусам, срокам
    - тренды изменения статуса
    - агрегации по пользователю (sk_id_curr)
    """
    df = df.copy()

    # Преобразование статуса в цифру
    mapping = {"C": 0, "X": 0, **{str(i): i for i in range(6)}}
    df["status_int"] = df["status"].map(mapping).fillna(0).astype(int)

    # 🎯 Фичи по кредиту (sk_bureau_id)
    def make_credit_level_features(df_):
        agg_months = df_.groupby("sk_bureau_id")["months_balance"].agg(
            bb_m_count="count",
            bb_m_min="min", bb_m_max="max",
            bb_m_mean="mean", bb_m_std="std"
        )

        agg_status = df_.groupby("sk_bureau_id")["status_int"].agg(
            bb_s_min="min", bb_s_max="max",
            bb_s_mean="mean", bb_s_std="std",
            bb_s_nunique="nunique",
            bb_good_months=lambda x: (x == 0).sum(),
            bb_bad_months=lambda x: (x > 0).sum()
        )

        def calc_slope(sub_):
            x = sub_["months_balance"].values.reshape(-1, 1)
            y = sub_["status_int"].values
            return float(LinearRegression().fit(x, y).coef_[0]) if len(x) > 1 else 0.0

        slope_df = df_.groupby("sk_bureau_id").apply(calc_slope).reset_index()
        slope_df.columns = ["sk_bureau_id", "bb_status_slope"]

        credit_feats = agg_months.merge(agg_status, on="sk_bureau_id")
        credit_feats = credit_feats.merge(slope_df, on="sk_bureau_id")
        return credit_feats.reset_index()

    credit_level = make_credit_level_features(df)

    # Базовая информация по кредиту (amt, date и т.д.)
    base = df[[
        "sk_bureau_id", "sk_id_curr", "days_credit", "days_enddate_fact",
        "amt_credit_sum", "amt_annuity", "credit_active"
    ]].drop_duplicates()

    merged = base.merge(credit_level, on="sk_bureau_id", how="left")

    # 🔁 Агрегация по клиенту (sk_id_curr)
    def make_client_level_features(df_):
        gb = df_.groupby("sk_id_curr")

        client = gb.agg(
            total_credits=("sk_bureau_id", "nunique"),
            active_credits=("credit_active", lambda x: (x == "Active").sum()),
            closed_credits=("credit_active", lambda x: (x != "Active").sum()),
            min_days_credit=("days_credit", "min"),
            max_days_credit=("days_credit", "max"),
            mean_days_credit=("days_credit", "mean"),
            std_days_credit=("days_credit", "std"),
            min_end_offset=("days_enddate_fact", "min"),
            max_end_offset=("days_enddate_fact", "max"),
            mean_end_offset=("days_enddate_fact", "mean"),
            std_end_offset=("days_enddate_fact", "std"),
            min_amt_credit=("amt_credit_sum", "min"),
            max_amt_credit=("amt_credit_sum", "max"),
            mean_amt_credit=("amt_credit_sum", "mean"),
            std_amt_credit=("amt_credit_sum", "std"),
            min_amt_annuity=("amt_annuity", "min"),
            max_amt_annuity=("amt_annuity", "max"),
            mean_amt_annuity=("amt_annuity", "mean"),
            std_amt_annuity=("amt_annuity", "std")
        ).reset_index()

        # Дополнительные BB-фичи (status, slope и т.п.)
        bb_cols = [col for col in df_.columns if col.startswith("bb_")]
        for col in bb_cols:
            client[f"{col}_min"] = gb[col].min().values
            client[f"{col}_max"] = gb[col].max().values
            client[f"{col}_mean"] = gb[col].mean().values
            client[f"{col}_std"] = gb[col].std().values
            client[f"{col}_sum"] = gb[col].sum().values

        # 👇 Инженерные признаки клиента
        client["pct_active"] = (
            client["active_credits"] / client["total_credits"].replace(0, np.nan)
        )
        client["early_close_ratio"] = (
            client["min_end_offset"] / client["max_days_credit"].replace(0, np.nan)
        )
        client["credit_annuity_ratio"] = (
            client["mean_amt_credit"] / client["mean_amt_annuity"].replace(0, np.nan)
        )
        client["dpd_heavy_ratio"] = (
            client["bb_bad_months_sum"] / client["bb_m_count_sum"].replace(0, np.nan)
        )
        client["status_slope_range"] = (
            client["bb_status_slope_max"] - client["bb_status_slope_min"]
        )
        client["balance_volatility"] = (
            client["bb_m_std_sum"] / client["bb_m_mean_sum"].replace(0, np.nan)
        )

        return client.reset_index(drop=True)

    client_feats = make_client_level_features(merged)
    return client_feats


def main():
    """
    Главная точка запуска скрипта:
    - загрузка данных из базы
    - объединение таблиц
    - извлечение фичей
    - сохранение результата
    """
    start = time.time()

    parser = argparse.ArgumentParser(description="Генерация bb-фичей")
    parser.add_argument("--output_file", required=True,
                        help="Путь к сохраняемому CSV-файлу")

    args = parser.parse_args()

    db = PostgresDBHandler(DB_ARGS)
    db.connect()

    df = load_data(db)
    features = generate_bb_features(df)

    features.to_csv(args.output_file, index=False)
    print(f"[✔] Сохранено: {args.output_file}")
    print(f"[⏱] Время выполнения: {time.time() - start:.2f} сек.")


if __name__ == "__main__":
    main()
