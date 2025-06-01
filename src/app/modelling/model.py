import argparse
import time
import warnings
import os
import sys
import pickle
import numpy as np
import pandas as pd
import shap

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import optuna

# Пути к кастомным модулям
sys.path.append(r"C:\Users\Denis\credit_scoring\src\app\utils")
sys.path.append(r"C:\Users\Denis\credit_scoring\src\config")
sys.path.append(r"C:\Users\Denis\credit_scoring\src\app\modelling\features")

from db_config import DB_ARGS
from db_handler import PostgresDBHandler

# Настройки
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)
np.set_printoptions(suppress=True, precision=2)

RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# 1. Загрузка и объединение данных
# ─────────────────────────────────────────────────────────────

def load_and_merge_data(application_file: str, bureau_balance_file: str):
    """
    Загружает данные из поданных CSV файлов, объединяет по sk_id_curr,
    делит на train/test, чистит пропуски.
    """
    print("[INFO] Загрузка данных...")

    # Подгрузка CSV-файлов
    df = pd.read_csv(application_file)
    bureau_balance = pd.read_csv(bureau_balance_file)

    # Делим на train/test по наличию target
    test = df[df["target"].isnull()].copy()
    train = df[df["target"].notnull()].copy()

    # Удаляем признаки с долей пропусков > 15%
    nan_thresh = 0.15
    missing_pct = train.isnull().mean()
    keep_cols = missing_pct[missing_pct <= nan_thresh].index.tolist()

    train = train[keep_cols]
    test = test[keep_cols]

    # Объединяем с данными bureau_balance
    train = train.merge(bureau_balance, on="sk_id_curr", how="left")
    test = test.merge(bureau_balance, on="sk_id_curr", how="left")

    # Заполняем оставшиеся пропуски -1
    train = train.fillna(-1)
    test["target"] = test["target"].fillna(999)
    test = test.fillna(-1)
    test["target"] = test["target"].replace(999, np.nan)

    return train, test


def preprocess_data(train: pd.DataFrame):
    """
    Делит фрейм на X и y, преобразует бинарные признаки.
    """
    print("[INFO] Предобработка...")

    X = train.drop(['sk_id_curr', 'target'], axis=1)
    y = train["target"].astype(int)

    # Преобразование бинарных признаков типа 'Y'/'N'
    binary_cols = ["flag_own_car", "flag_own_realty"]
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].replace({'Y': 1, 'N': 0})

    return X, y

# ─────────────────────────────────────────────────────────────
# 3. Построение пайплайна
# ─────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(X):
    """
    Строит ColumnTransformer с числовыми, категориальными и бинарными признаками.
    """
    binary_cols = [col for col in X.columns if X[col].nunique() == 2 and X[col].dtype in ['int', 'float']]

    num_cols = [col for col in X.select_dtypes(include=['int', 'float']).columns
                if col not in binary_cols]

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ("bin", "passthrough", binary_cols)
    ])

    return preprocessor

# ─────────────────────────────────────────────────────────────
# 4. Фильтрация признаков по SHAP
# ─────────────────────────────────────────────────────────────

def select_top_shap_features(X, y, top_n=100):
    """
    Обучает LogisticRegression и возвращает топ-N признаков по SHAP важности
    """
    print("[INFO] Выбор признаков по SHAP...")

    preprocessor = build_preprocessing_pipeline(X)
    model = LogisticRegression(random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Делим на обучающую и валидационную выборки
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    X_train_proc = pipeline.named_steps['preprocessor'].transform(X_train)
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    explainer = shap.LinearExplainer(pipeline.named_steps["model"], X_train_proc)
    shap_values = explainer.shap_values(X_train_proc)

    shap_importances = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)

    top_features = shap_importances.head(top_n).index.tolist()
    print(f"[INFO] Топ-{top_n} признаков выбрано.")
    return top_features

# ─────────────────────────────────────────────────────────────
# 5. Гиперпараметры и обучение
# ─────────────────────────────────────────────────────────────

def optimize_and_evaluate(X, y, selected_features):
    """
    Проводит Optuna оптимизацию для логистической регрессии и дерева решений
    только на выбранных фичах.
    """
    X = X[selected_features].copy()

    def objective_logreg(trial):
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)

        preprocessor = build_preprocessing_pipeline(X)
        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
        return -np.mean(score)

    def objective_dt(trial):
        max_depth = trial.suggest_int('max_depth', 2, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 100)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 100)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

        preprocessor = build_preprocessing_pipeline(X)
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=42
        )
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        score = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
        return -np.mean(score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Оптимизация логрег
    print("[INFO] Optuna: Logistic Regression")
    study_lr = optuna.create_study(direction='minimize')
    study_lr.optimize(objective_logreg, n_trials=20, n_jobs=2)

    print("[INFO] Optuna: Decision Tree")
    study_dt = optuna.create_study(direction='minimize')
    study_dt.optimize(objective_dt, n_trials=20, n_jobs=2)

    # Обучение победителя
    best_lr = LogisticRegression(**study_lr.best_params, random_state=42)
    best_dt = DecisionTreeClassifier(**study_dt.best_params, random_state=42)
    preproc = build_preprocessing_pipeline(X)

    pipe_lr = Pipeline([('preprocessor', preproc), ('model', best_lr)])
    pipe_dt = Pipeline([('preprocessor', preproc), ('model', best_dt)])

    score_lr = cross_val_score(pipe_lr, X, y, cv=5, scoring='roc_auc')
    score_dt = cross_val_score(pipe_dt, X, y, cv=5, scoring='roc_auc')

    print(f"LogReg AUC:  {np.mean(score_lr):.4f}")
    print(f"DecTree AUC: {np.mean(score_dt):.4f}")

    best_model = pipe_dt if np.mean(score_dt) > np.mean(score_lr) else pipe_lr
    model_name = "decision_tree" if best_model is pipe_dt else "logreg"

    with open(f"src/app/modelling/final_model_{model_name}.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"[✅] Финальная модель сохранена как final_model_{model_name}.pkl")


# ─────────────────────────────────────────────────────────────
# 6. Точка входа
# ─────────────────────────────────────────────────────────────
    
def main():
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Обучение модели с SHAP и Optuna"
    )
    parser.add_argument("--application_file", required=True,
                        help="Файл с признаками application_train_test")
    parser.add_argument("--bureau_balance_file", required=True,
                        help="Файл с признаками bureau_balance")
    parser.add_argument("--output_file", required=True,
                        help="Путь для сохранения SHAP важностей")

    args = parser.parse_args()

    # Загрузка объединённых данных
    train, _ = load_and_merge_data(args.application_file, args.bureau_balance_file)

    # Предобработка
    X, y = preprocess_data(train)

    # SHAP фильтрация → топовые признаки
    selected_features = select_top_shap_features(X, y, top_n=100)

    # Сохраняем SHAP фичи
    pd.DataFrame({'feature': selected_features}).to_csv(args.output_file, index=False)
    print(f"[📈] SHAP признаки сохранены: {args.output_file}")

    # Оптимизация и финальное обучение
    optimize_and_evaluate(X, y, selected_features)

    print(f"[⏱] Время выполнения: {time.time() - start:.2f} сек.")
