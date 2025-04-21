import os, argparse, joblib, mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/WA_Fn-UseC-Telco-Customer-Churn.csv"
PROCESSED_PATH  = "data/processed/train_processed.csv"
PREPROCESS_PKL  = "models/preprocess.pkl"

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_transformer(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if "Churn" in numeric_cols: numeric_cols.remove("Churn")
    if "Churn" in categorical_cols: categorical_cols.remove("Churn")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformer = ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])

    return transformer

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])
    transformer = build_transformer(df)
    X_prep = transformer.fit_transform(X)

    feature_names = (
        transformer.get_feature_names_out()
        if hasattr(transformer, "get_feature_names_out")
        else None
    )
    X_prep_df = pd.DataFrame(X_prep, columns=feature_names)
    X_prep_df["Churn"] = y.values
    return X_prep_df, transformer

def save_outputs(df: pd.DataFrame, transformer):
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PREPROCESS_PKL), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    joblib.dump(transformer, PREPROCESS_PKL)

def main():
    mlflow.set_experiment("churn-preprocess")
    with mlflow.start_run(run_name="v1-preprocess"):
        df_raw = load_raw(RAW_PATH)
        df_proc, transf = preprocess(df_raw)
        save_outputs(df_proc, transf)

        mlflow.log_artifact(PROCESSED_PATH, artifact_path="processed_data")
        mlflow.log_artifact(PREPROCESS_PKL, artifact_path="preprocess_model")
        mlflow.log_param("rows", len(df_raw))
        mlflow.log_metric("missing_rate", df_raw.isna().mean().mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=RAW_PATH)
    parser.add_argument("--output", default=PROCESSED_PATH)
    args = parser.parse_args()
    main()
