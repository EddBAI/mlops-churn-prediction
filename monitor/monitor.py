import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from datetime import datetime

train_path = "data/raw/WA_Fn-UseC-Telco-Customer-Churn.csv"
new_data_path = "data/processed/new_data.csv"

df_train = pd.read_csv(train_path)

df_new = pd.read_csv(new_data_path)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_train, current_data=df_new)

data_hora = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_path = f"monitor/drift_report_{data_hora}.html"
report.save_html(output_path)

print(f"Relatório Evidently gerado em: {output_path}")