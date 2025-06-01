"""
Модуль для обучения моделей LightGBM и RandomForest с использованием Optuna
и SHAP-фильтрацией признаков. Поддерживает использование в DVC-пайплайне.
"""

import os
import sys
import time
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import shap
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


RANDOM_STATE = 42
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. Загрузка данных
# ─────────────────────────────────────────────────────────────

def load_and_merge_data(application_path: str, bureau_path: str):
    """
    Загружает данные из двух файлов (application + bureau_balance) и объединяет
    """
    print("[INFO] Загрузка и объединение данных...")

    bureau_balance = pd.read_csv(bureau_path)
    df = pd.read_csv(application_path)

    test = df[df["target"].isnull()].copy()
    train = df[df["target"].notnull()].copy()

    nan_thresh = 0.15
    valid_cols = (train.isnull().mean() <= nan_thresh).index.tolist()
    train = train[valid_cols]
    test = test[valid_cols]

    train = train.merge(bureau_balance, on="sk_id_curr", how="left")
    test = test.merge(bureau_balance, on="sk_id_curr", how="left")

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    test["target"] = np.nan

    return train, test

# ─────────────────────────────────────────────────────────────
# 2. Подготовка данных
# ─────────────────────────────────────────────────────────────

def preprocess_data(train):
    """
    Делит train на X/y и обрабатывает бинарные признаки
    """
    print("[INFO] Предобработка...")
    X = train.drop(['target', 'sk_id_curr'], axis=1)
    y = train['target'].astype(int)

    # Преобразование Y/N в 0/1
    for col in ["flag_own_car", "flag_own_realty"]:
        if col in X.columns:
            X[col] = X[col].replace({'Y': 1, 'N': 0})

    return X, y

# ─────────────────────────────────────────────────────────────
# 3. Препроцессор
# ─────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(X):
    """
    Возвращает ColumnTransformer с обработкой числовых, бинарных и категориальных столбцов.
    """
    bin_cols = [col for col in X.columns if X[col].nunique() == 2]
    num_cols = list(set(X.select_dtypes(include=["int", "float"]).columns) - set(bin_cols))
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        X[cat_cols] = X[cat_cols].astype(str)

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", bin_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

# ─────────────────────────────────────────────────────────────
# 4. SHAP-фильтрация признаков
# ─────────────────────────────────────────────────────────────

def select_top_shap_features(X, y, top_n=100):
    """
    Обучает базовую модель, вычисляет SHAP и оставляет топ-N признаков
    """
    print("[INFO] Отбор признаков по SHAP...")

    preprocessor = build_preprocessing_pipeline(X)
    model = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    X_transformed = pipeline.named_steps["preprocessor"].transform(X_val)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    explainer = shap.Explainer(pipeline.named_steps["model"])
    shap_values = explainer(X_transformed)

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values(by="importance", ascending=False)

    top_features = shap_df.head(top_n)["feature"].tolist()
    return top_features, shap_df

# ─────────────────────────────────────────────────────────────
# 5. Оптимизация и обучение модели
# ─────────────────────────────────────────────────────────────

def optimize_and_train(X, y, selected_features, output_dir="src/app/modelling"):
    """
    Оптимизация и обучение моделей LightGBM и RandomForest по отобранным признакам
    """
    X = X.copy()
    X = X[selected_features]

    def objective_lgb(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": RANDOM_STATE
        }
        preproc = build_preprocessing_pipeline(X)
        model = lgb.LGBMClassifier(**params, n_estimators=500)
        pipe = Pipeline([("preprocessor", preproc), ("model", model)])
        score = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        return -np.mean(score)

    def objective_rf(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        preproc = build_preprocessing_pipeline(X)
        model = RandomForestClassifier(**params)
        pipe = Pipeline([("preprocessor", preproc), ("model", model)])
        score = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
        return -np.mean(score)

    print("[INFO] Оптимизация LightGBM")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=20, n_jobs=2)

    print("[INFO] Оптимизация RandomForest")
    study_rf = optuna.create_study(direction="minimize")
    study_rf.optimize(objective_rf, n_trials=20, n_jobs=2)

    # Оценка результатов
    best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, n_estimators=500, random_state=RANDOM_STATE)
    best_rf = RandomForestClassifier(**study_rf.best_params, class_weight="balanced", random_state=RANDOM_STATE)

    pipe_lgb = Pipeline([("preprocessor", build_preprocessing_pipeline(X)), ("model", best_lgb)])
    pipe_rf = Pipeline([("preprocessor", build_preprocessing_pipeline(X)), ("model", best_rf)])

    auc_lgb = cross_val_score(pipe_lgb, X, y, cv=10, scoring="roc_auc").mean()
    auc_rf = cross_val_score(pipe_rf, X, y, cv=10, scoring="roc_auc").mean()

    print(f"LightGBM AUC: {auc_lgb:.4f}")
    print(f"RandomForest AUC: {auc_rf:.4f}")

    best_pipe = pipe_lgb if auc_lgb > auc_rf else pipe_rf
    best_name = "lightgbm" if auc_lgb > auc_rf else "randomforest"

    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(best_pipe, open(f"{output_dir}/final_model_{best_name}.pkl", "wb"))
    print(f"[✅] Модель сохранена как final_model_{best_name}.pkl")


# ─────────────────────────────────────────────────────────────
# 6. Основная функция
# ─────────────────────────────────────────────────────────────

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--application_file", required=True,
                        help="Путь до features_application_train_test.csv")
    parser.add_argument("--bureau_balance_file", required=True,
                        help="Путь до features_bureau_balance.csv")
    parser.add_argument("--output_file", required=True,
                        help="Путь для сохранения SHAP важностей")
    args = parser.parse_args()

    train, _ = load_and_merge_data(args.application_file, args.bureau_balance_file)
    X, y = preprocess_data(train)

    top_features, shap_df = select_top_shap_features(X, y, top_n=100)
    shap_df.to_csv(args.output_file, index=False)
    print(f"[📈] SHAP важности сохранены: {args.output_file}")

    optimize_and_train(X, y, selected_features=top_features)

    print(f"[⏱] Время выполнения: {time.time() - start:.2f} сек.")


if __name__ == "__main__":
    main()
