import requests
import joblib
import os
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
import random
import numpy as np
import os
import warnings

load_dotenv()

def get_model():
    model_path = "model/rf_uas_model.pkl"
    
    # Cek jika model sudah ada di lokal untuk menghemat bandwidth
    if os.path.exists(model_path):
        return joblib.load(model_path)

    # Autentikasi (Input token jika repo private)
    dagshub_username = os.getenv("DAGSHUB_USERNAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    uri_artifacts = "https://dagshub.com/AlfanoIhz/uas-dsp.mlflow"
    mlflow.set_tracking_uri(uri_artifacts)
    
    model_uri = "models:/rf_uas_model/2" 
    model = mlflow.sklearn.load_model(model_uri)
    
    if not os.path.exists('model'):
        os.makedirs('model')
        
    joblib.dump(model, model_path)
    return model

def generate_random_features():
    """Menghasilkan data acak sesuai rentang dataset Job Change"""
    return {
        "city": random.randint(0, 140),
        "city_development_index": round(random.uniform(0.4, 0.95), 3),
        "gender": random.randint(0, 2),
        "relevent_experience": random.randint(0, 1),
        "enrolled_university": random.randint(0, 2),
        "education_level": random.randint(0, 4),
        "major_discipline": random.randint(0, 5),
        "experience": random.randint(0, 22),
        "company_size": random.randint(0, 7),
        "company_type": random.randint(0, 5),
        "last_new_job": random.randint(0, 4),
        "training_hours": random.randint(1, 336)
    }

# Load encoders secara global agar efisien
# Pastikan file ini sudah ada di folder model/
def load_encoders():
    try:
        edu_map = joblib.load('model/education_map.pkl')
        company_encoder = joblib.load('model/label_encoder.pkl')
        return edu_map, company_encoder
    except:
        return None, None

def apply_preprocessing(input_data):
    """
    Mengubah data dari form (string) menjadi angka menggunakan encoder.
    input_data: dict dari request.form
    """
    edu_map, company_encoder = load_encoders()
    
    # Salin data agar tidak merusak original
    processed_data = input_data.copy()

    # 1. Transformasi Education Level menggunakan Map
    if edu_map and 'education_level' in processed_data:
        # Jika user input string, petakan ke angka. Jika tidak ada di map, beri 0
        val = processed_data['education_level']
        processed_data['education_level'] = edu_map.get(val, 0)

    # 2. Transformasi Company Type menggunakan LabelEncoder
    if company_encoder and 'company_type' in processed_data:
        try:
            val = processed_data['company_type']
            # LabelEncoder sklearn butuh input berupa list/array
            res = company_encoder.transform([val])[0]
            processed_data['company_type'] = res
        except:
            processed_data['company_type'] = 0 # Default jika kategori baru

    # Pastikan semua kolom lain dikonversi ke numerik (float/int)
    for key in processed_data:
        processed_data[key] = float(processed_data[key])
        
    return processed_data

def load_model():
    MODEL_PATH = "model/rf_uas_model.pkl"
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    
    
def set_features():
    """Daftar fitur sesuai urutan X_train di notebook UAS Anda"""
    return [
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