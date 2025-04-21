import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from datetime import datetime

# Caminhos
RAW_DATA_PATH = "data/raw/WA_Fn-UseC-Telco-Customer-Churn.csv"
PREPROCESSOR_PATH = "models/preprocess.pkl"

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Remove colunas irrelevantes
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df = df[df["TotalCharges"].notna()]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

from sklearn.impute import SimpleImputer

def build_transformer(df):
    num_cols = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Churn"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformer = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return transformer


def train_and_log_models(X_train, X_test, y_train, y_test, experiment_name="churn-retrain"):

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = -1

    with mlflow.start_run(run_name="Retrain_" + datetime.now().strftime("%Y-%m-%d_%H-%M")):
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred)

            mlflow.log_metric(f"f1_{name}", score)
            mlflow.sklearn.log_model(model, artifact_path=name)

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_name}"
        mlflow.register_model(model_uri, "ChurnModel")
        print(f"Modelo '{best_name}' registrado no MLflow Registry com F1: {best_score:.4f}")

def main():
    print("Iniciando re-treinamento do modelo...")

    df = load_and_clean_data(RAW_DATA_PATH)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    transformer = build_transformer(df)
    X_transformed = transformer.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(transformer, PREPROCESSOR_PATH)
    print(f"Novo transformer salvo em: {PREPROCESSOR_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, stratify=y, random_state=42)
    train_and_log_models(X_train, X_test, y_train, y_test)

    print("Fim.")

if __name__ == "__main__":
    main()
