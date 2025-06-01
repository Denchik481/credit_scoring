"""Модуль для работы с базой данных PostgreSQL."""

import pandas as pd
import psycopg2


class PostgresDBHandler:
    """
    Класс для работы с базой данных PostgreSQL.

    Предоставляет методы для выполнения SQL-запросов, создания таблиц,
    загрузки данных и управления связями между таблицами.
    """

    def __init__(self, db_args: dict):
        """
        Инициализирует объект, считывая параметры подключения из переданного словаря db_args.

        :param db_args: Словарь с параметрами подключения (host, port, dbname, user, password).
        """
        self.host = db_args.get("host", "localhost")
        self.port = db_args.get("port", 5432)
        self.dbname = db_args.get("dbname", "default_db")
        self.user = db_args.get("user", "default_user")
        self.password = db_args.get("password", "default_password")
        self.connection = None

    def connect(self):
        """Устанавливает соединение с PostgreSQL."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
            )
            print("Соединение с PostgreSQL установлено.")
        except psycopg2.Error as error:
            print("Ошибка при установке соединения:", error)

    def close(self):
        """Закрывает соединение с базой данных."""
        if self.connection:
            self.connection.close()
            print("Соединение закрыто.")

    def execute_query(self, query: str, params: tuple = None):
        """
        Выполняет произвольный SQL-запрос (DDL или DML).

        :param query: SQL-запрос.
        :param params: параметры запроса (если требуются).
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, params)
                self.connection.commit()
        except (Exception, psycopg2.Error) as e:
            print("Ошибка при выполнении запроса:", e)

    def get_df_from_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """
        Выполняет SELECT-запрос и возвращает результат в виде DataFrame.

        :param query: SQL-запрос.
        :param params: параметры запроса.
        :return: DataFrame с результатами или None в случае ошибки.
        """
        try:
            df = pd.read_sql(query, self.connection, params=params)
            return df
        except Exception as e:
            print("Ошибка при получении данных:", e)
            return None

    def create_table(
        self, table_name: str, columns: dict, drop_if_exists: bool = False
    ):
        """
        Создает новую таблицу с заданными столбцами и типами.

        :param table_name: имя таблицы.
        :param columns: словарь, где ключ – имя столбца, а значение – тип столбца.
                        Например: {"id": "SERIAL PRIMARY KEY", "name": "TEXT NOT NULL"}
        :param drop_if_exists: если True, удалит таблицу, если она существует.
        """
        try:
            if drop_if_exists:
                self.execute_query(f"DROP TABLE IF EXISTS {table_name};")
                print(f"Таблица {table_name} удалена (если существовала).")

            cols = ", ".join([f"{col} {col_type}" for col, col_type in columns.items()])
            self.execute_query(f"CREATE TABLE {table_name} ({cols});")
            print(f"Таблица {table_name} успешно создана.")
        except Exception as error:
            print("Ошибка при создании таблицы:", error)

    def load_table_from_csv(
        self,
        table_name: str,
        csv_file_path: str,
        delimiter: str = ",",
        header: bool = True,
    ):
        """
        Загружает данные из локального CSV-файла в указанную таблицу.

        :param table_name: имя таблицы.
        :param csv_file_path: путь к CSV-файлу.
        :param delimiter: разделитель (по умолчанию ',').
        :param header: наличие заголовка (по умолчанию True).
        """
        copy_sql = f"""
        COPY {table_name}
        FROM STDIN WITH CSV {'HEADER' if header else ''} DELIMITER '{delimiter}';
        """
        try:
            with (
                self.connection.cursor() as cursor,
                open(csv_file_path, "r", encoding="utf-8") as file,
            ):
                cursor.copy_expert(copy_sql, file)
            self.connection.commit()
            print(f"Данные успешно загружены в таблицу {table_name}.")
        except (Exception, psycopg2.Error) as error:
            print("Ошибка при загрузке данных из CSV:", error)

    def establish_relationship(
        self,
        table_name: str,
        column_name: str,
        ref_table: str,
        ref_column: str,
        constraint_name: str,
    ):
        """
        Устанавливает внешний ключ (связь между таблицами) в указанной таблице.

        :param table_name: таблица, в которой создается внешний ключ.
        :param column_name: столбец, который станет внешним ключом.
        :param ref_table: таблица-источник.
        :param ref_column: столбец в таблице-источнике (обычно первичный или уникальный).
        :param constraint_name: имя ограничения для внешнего ключа.
        """
        alter_sql = (
            f"ALTER TABLE {table_name} "
            f"ADD CONSTRAINT {constraint_name} "
            f"FOREIGN KEY ({column_name}) "
            f"REFERENCES {ref_table}({ref_column});"
        )
        try:
            self.execute_query(alter_sql)
            print(
                f"Внешняя связь {table_name}.{column_name} -> {ref_table}.{ref_column} установлена."
            )
        except Exception as error:
            print("Ошибка при установке внешней связи:", error)

    def establish_primary_key(
        self, table_name: str, columns: list, constraint_name: str = None
    ):
        """
        Устанавливает первичный ключ для таблицы.
        Если constraint_name не указан, генерируется по умолчанию.

        :param table_name: имя таблицы.
        :param columns: список столбцов, образующих первичный ключ.
        :param constraint_name: имя ограничения для первичного ключа.
        """
        if not constraint_name:
            constraint_name = f"pk_{table_name}"
        cols = ", ".join(columns)
        alter_sql = f"""
        ALTER TABLE {table_name}
        ADD CONSTRAINT {constraint_name} PRIMARY KEY ({cols});
        """
        try:
            self.execute_query(alter_sql)
            print(
                f"Первичный ключ для таблицы {table_name} установлен на столбцы ({cols})."
            )
        except Exception as e:
            print("Ошибка при установке первичного ключа:", e)
