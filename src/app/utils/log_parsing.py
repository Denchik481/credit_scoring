"""Модуль для обработки логов и анализа данных."""

import argparse
import ast
import json
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class PosCashBalanceIDs:
    """Класс для хранения данных POS_CASH_balance."""

    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str


@dataclass
class AmtCredit:
    """Класс для хранения данных о кредите."""

    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float


def read_log_file(log_file_path):
    """
    Читает .log файл и разделяет данные на две категории: bureau и POS_CASH_balance.

    :param log_file_path: Путь к файлу логов.
    :return: Кортеж из двух списков: bureau и POS_CASH_balance.
    """
    bureau = []
    pos_cash_balance = []
    with open(log_file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row["type"] == "bureau":
                bureau.append(row["data"])
            elif row["type"] == "POS_CASH_balance":
                pos_cash_balance.append(row["data"])
    return bureau, pos_cash_balance


def process_bureau_data(bureau_data):
    """
    Обрабатывает данные bureau и возвращает DataFrame.

    :param bureau_data: Список данных bureau.
    :return: DataFrame с обработанными данными.
    """
    df = pd.DataFrame(bureau_data)
    df_normalized = pd.json_normalize(df["record"])
    df = pd.concat([df, df_normalized], axis=1).drop(columns=["record"])
    df["AmtCredit"] = df["AmtCredit"].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value
    )
    df["AmtCredit_dict"] = df["AmtCredit"].apply(asdict)
    amtcredit_df = pd.json_normalize(df["AmtCredit_dict"])
    df = pd.concat(
        [df.drop(columns=["AmtCredit", "AmtCredit_dict"]), amtcredit_df], axis=1
    )
    return df


def process_pos_cash_balance_data(pos_cash_balance_data):
    """
    Обрабатывает данные POS_CASH_balance и возвращает DataFrame.

    :param pos_cash_balance_data: Список данных POS_CASH_balance.
    :return: DataFrame с обработанными данными.
    """
    df = pd.DataFrame(pos_cash_balance_data)
    df = df.explode("records").reset_index(drop=True)
    df_norm = pd.json_normalize(df["records"])
    df = pd.concat([df.drop(columns=["records"]), df_norm], axis=1)
    df["PosCashBalanceIDs"] = df["PosCashBalanceIDs"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["PosCashBalanceIDs_dict"] = df["PosCashBalanceIDs"].apply(lambda x: asdict(x))
    pos_ids = pd.json_normalize(df["PosCashBalanceIDs_dict"])
    df = pd.concat(
        [df.drop(columns=["PosCashBalanceIDs", "PosCashBalanceIDs_dict"]), pos_ids],
        axis=1,
    )
    return df


def save_to_csv(dataframe, output_path):
    """
    Сохраняет DataFrame в .csv файл.

    :param dataframe: DataFrame для сохранения.
    :param output_path: Путь для сохранения файла.
    """
    dataframe.to_csv(output_path, index=False)


def main():
    """Основная функция для обработки .log файла и сохранения данных в .csv файлы."""

    parser = argparse.ArgumentParser(
        description="Обработка .log файла и сохранение в .csv файлы."
    )
    parser.add_argument("--log_file", required=True, help="Путь к входному .log файлу")
    parser.add_argument(
        "--bureau_csv", required=True, help="Путь к выходному .csv файлу для bureau"
    )
    parser.add_argument(
        "--pos_cash_balance_csv",
        required=True,
        help="Путь к выходному .csv файлу для POS_CASH_balance",
    )
    args = parser.parse_args()

    bureau_data, pos_cash_balance_data = read_log_file(args.log_file)

    bureau_df = process_bureau_data(bureau_data)
    pos_cash_balance_df = process_pos_cash_balance_data(pos_cash_balance_data)

    save_to_csv(bureau_df, args.bureau_csv)
    save_to_csv(pos_cash_balance_df, args.pos_cash_balance_csv)

    print("Обработка завершена. Данные сохранены в .csv файлы.")


if __name__ == "__main__":
    main()
