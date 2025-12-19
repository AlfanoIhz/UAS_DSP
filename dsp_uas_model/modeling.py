import os
import warnings
import sys
import pandas as pd
import numpy as np
import dagshub
import mlflow
import joblib
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
from mlflow.client import MlflowClient

warnings.filterwarnings("ignore")
load_dotenv()

def run_rf_model_mlflow(df):
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    url_dagshub = "https://dagshub.com/AlfanoIhz/uas-dsp.mlflow"

    mlflow.set_tracking_uri(url_dagshub)
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    # Set mlflow
    experiment_name = "uas_dsp"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = None
    client = MlflowClient()
    if experiment is None:
        experiment_id = client.create_experiment(name = experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
        experiment = mlflow.get_experiment(experiment_id)
    else:
        print(f"Experiment '{experiment_name}' already exists")

    # Set Data Preparation
    selected_features = [
        "city", 
        "city_development_index", 
        "gender", 
        "relevent_experience",
        "enrolled_university", 
        "education_level", 
        "major_discipline",
        "experience", 
        "company_size", 
        "company_type", 
        "last_new_job", 
        "training_hours"
    ]
    
    y = df['target']
    X = df[selected_features]  # Hanya gunakan features yang sama dengan app
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="rf-uas-model", experiment_id=experiment.experiment_id) as run:
        
        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)
        y_pred = model_rf.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # Logging parameter
        model_params = model_rf.get_params()
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"tuned_{param_name}", param_value)

        # Logging metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)

        # Simpan model dengan joblib (kompatibel dengan DagsHub)
        model_filename = "rf_uas_model.pkl"
        joblib.dump(model_rf, model_filename)
        
        # Log model ke MLflow dan DagsHub Model Registry
        mlflow.sklearn.log_model(
            sk_model=model_rf, 
            artifact_path="model", # Folder di cloud
            registered_model_name="rf_uas_model" # Nama di Registry
        )

        print(f"Run ID: {run.info.run_id}")
        
        # Hapus file lokal setelah upload
        os.remove(model_filename)

        # Print run info
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))

if __name__ == "__main__":
    dataset = "data/final_clean.csv"
    df = pd.read_csv(dataset)
    run_rf_model_mlflow(df)