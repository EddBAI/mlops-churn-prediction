"""
train.py
========
Treina dois algoritmos, compara métricas e registra o melhor no MLflow Registry.
Uso:
    python src/models/train.py --data data/processed/train_processed.csv
"""

import argparse, os, joblib, mlflow, mlflow.sklearn, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime

RANDOM_STATE = 42
TARGET = "Churn"
EXPERIMENT_NAME = "churn-training"

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

def get_models():
    return {
        "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "gbt": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

def eval_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_proba)
    }

def main(data_path: str):
    # ------------------------------------------------------------------ MLflow
    mlflow.set_tracking_uri("mlruns")          # pasta local (SQLite nativo)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data(data_path)
    best_auc, best_run_id = 0.0, None

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_prob  = model.predict_proba(X_test)[:, 1]

            metrics = eval_metrics(y_test, y_pred, y_prob)
            mlflow.log_metrics(metrics)

            # Salva hiperparâmetros simples
            mlflow.log_params(model.get_params())

            # Loga artefato + assinatura
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="ChurnModel"
            )

            # Mantém controle do melhor
            if metrics["roc_auc"] > best_auc:
                best_auc, best_run_id = metrics["roc_auc"], run.info.run_id

    # ------------------------------------------------------- promoção automática
    if best_run_id:
        client = mlflow.tracking.MlflowClient()
        mv = client.get_latest_versions(
            name="ChurnModel", stages=["None"]
        )[0]   # mais recente sem estágio
        client.transition_model_version_stage(
            name="ChurnModel",
            version=mv.version,
            stage="Staging",
            archive_existing_versions=True
        )
        print(f"✔️  Versão {mv.version} promovida a STAGING (run {best_run_id})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="data/processed/train_processed.csv",
                        help="caminho do CSV pré‑processado")
    args = parser.parse_args()
    main(args.data)
