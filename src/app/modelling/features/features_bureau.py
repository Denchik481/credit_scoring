import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Пути к локальным модулям
sys.path.append(r"C:\Users\Denis\credit_scoring\src\app\utils")
sys.path.append(r"C:\Users\Denis\credit_scoring\src\config")

from db_config import DB_ARGS
from db_handler import PostgresDBHandler

# Отключаем предупреждения пакета
warnings.filterwarnings("ignore")


def load_bureau_data(db_handler: PostgresDBHandler) -> pd.DataFrame:
    """
    Загружает выборку из таблицы bureau с нужными колонками.
    """
    cols = [
        "sk_id_curr", "sk_bureau_id", "amt_credit_max_overdue",
        "amt_credit_sum_overdue", "amt_credit_sum",
        "credit_type", "credit_day_overdue", "credit_active"
    ]
    query = f"SELECT {', '.join(cols)} FROM bureau"
    return db_handler.get_df_from_query(query)


def preprocess_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка данных bureau:
    - заполнение пропусков по кредиту и просрочкам
    - типизация числовых и категориальных признаков
    """
    df = df.copy()

    # Заполнение пропусков
    df["amt_credit_max_overdue"] = df["amt_credit_max_overdue"].fillna(0)
    df["amt_credit_sum_overdue"] = df["amt_credit_sum_overdue"].fillna(0)
    df["amt_credit_sum"] = df["amt_credit_sum"].fillna(1)

    # Типизация
    for col in ["credit_type", "credit_active"]:
        df[col] = df[col].astype("category")

    for col in ["amt_credit_max_overdue", "amt_credit_sum", "amt_credit_sum_overdue"]:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def generate_bureau_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация признаков по таблице bureau:
    - агрегации по клиенту
    - share-метрики по активным займам
    - информация по типам кредитов
    """
    df = df.copy()

    # === Признаки по просрочкам каждого клиента ============================
    df["max_overdue_sum"] = df.groupby("sk_id_curr")["amt_credit_sum_overdue"].transform("max")
    df["min_overdue_sum"] = df.groupby("sk_id_curr")["amt_credit_sum_overdue"].transform("min")

    # Только активные займы
    active_df = df[df["credit_active"] == "Active"].copy()
    active_df["overdue_share"] = (
        active_df["amt_credit_sum_overdue"] /
        active_df["amt_credit_sum"].replace(0, np.nan)
    )

    overdue_stats = (
        active_df.groupby("sk_id_curr")["overdue_share"]
        .agg(
            mean_overdue_share="mean",
            max_overdue_share="max",
            sum_overdue_share="sum",
            count_overdue_loans=lambda x: (x > 0).sum()
        )
        .reset_index()
    )

    # === Агрегации по типу кредита (credit_type) ============================
    credit_type_stats = (
        df.groupby("credit_type")["sk_bureau_id"]
        .count()
        .reset_index(name="total_credits_count")
        .merge(
            df.loc[df["amt_credit_sum_overdue"] > 0]
            .groupby("credit_type")["sk_bureau_id"]
            .count()
            .reset_index(name="overdue_credits_count"),
            on="credit_type", how="left"
        )
        .merge(
            df.loc[df["credit_active"] == "Closed"]
            .groupby("credit_type")["sk_bureau_id"]
            .count()
            .reset_index(name="closed_credits_count"),
            on="credit_type", how="left"
        )
        .fillna(0)
    )

    # === Объединение всех признаков с основной таблицей ====================
    df = df.merge(overdue_stats, on="sk_id_curr", how="left")
    df = df.merge(credit_type_stats, on="credit_type", how="left")

    return df


def main():
    """
    Основная функция:
    - загрузка данных
    - предобработка
    - генерация признаков
    - сохранение в .csv
    """
    start_time = time.time()

    # Параметры запуска
    parser = argparse.ArgumentParser(
        description="Генерация признаков на основе таблицы bureau"
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Путь к выходному файлу с признаками (CSV)"
    )
    args = parser.parse_args()

    # Соединение с БД
    db = PostgresDBHandler(DB_ARGS)
    db.connect()

    # Загрузка и подготовка
    bureau_raw = load_bureau_data(db)
    bureau_clean = preprocess_bureau(bureau_raw)

    # Генерация признаков
    bureau_features = generate_bureau_features(bureau_clean)

    # Сохранение результата
    bureau_features.to_csv(args.output_file, index=False)
    print(f"[✅] Файл успешно сохранён: {args.output_file}")
    print(f"[⏱] Время выполнения скрипта: {time.time() - start_time:.2f} сек.")


if __name__ == "__main__":
    main()