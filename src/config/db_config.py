"""Конфигурация базы данных."""

import os
from types import MappingProxyType

DB_ARGS = MappingProxyType(
    {
        "host": "localhost",
        "port": 5432,
        "dbname": "home_credit",
        "user": "postgres",
        "password": os.getenv("DB_PASSWORD", "default_password"),
    },
)
