from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib, mlflow.pyfunc, os

TRANSFORMER_PATH = "models/preprocess.pkl"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_URI  = "models:/ChurnModel/Staging" 

try:
    transformer = joblib.load(TRANSFORMER_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Transformer não encontrado em {TRANSFORMER_PATH}") from e

mlflow.set_tracking_uri(MLFLOW_URI)
# model = mlflow.pyfunc.load_model(MODEL_URI)
from mlflow.sklearn import load_model
model = load_model(MODEL_URI) 

app = FastAPI(title="Telco Churn API", version="0.1.0")

@app.get("/")
def healthcheck():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: dict):
    try:
        df_raw = pd.DataFrame([customer])
        df_proc = transformer.transform(df_raw)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_proc)[:, 1][0]   
        else:
            proba = float(model.predict(df_proc)[0])
        churn_cls = proba >= 0.5

        return {
            "churn_probability": round(proba, 4),
            "churn": bool(churn_cls)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro de inferência: {e}")