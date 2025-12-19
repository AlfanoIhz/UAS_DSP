from flask import Flask, render_template, request, jsonify
from model_util import generate_random_features, get_model, load_model, set_features, apply_preprocessing 
import joblib
import pandas as pd
import os

# from model_util import (
#     generate_random_features, get_model, load_model, 
#     set_features, apply_preprocessing 
# )

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def get_dashboard():
    return render_template('dashboard.html')

@app.route('/model')
def get_model_new():
    model = get_model() # Memanggil fungsi yang diperbaiki di atas
    if model:
        status = "Model berhasil dimuat!"
    else:
        status = "Gagal memuat model. Periksa koneksi atau kredensial DagsHub."
    return render_template('predict_view.html', status=status)


@app.route('/predict', methods=['GET', 'POST'])
def predict_view():
    features_names = set_features()
    model = load_model()
    features = generate_random_features() if request.method == "GET" else None
    prediction = None

    if request.method == "POST":
        # 1. Ambil data mentah dari Form
        raw_features = {name: request.form.get(name, 0) for name in features_names}
        
        # 2. Jalankan Preprocessing (Memastikan urutan dan tipe data benar)
        processed_features = apply_preprocessing(raw_features)
        
        # 3. Ubah ke DataFrame dengan URUTAN KOLOM yang BENAR
        # PENTING: Urutan kolom harus sama dengan saat training
        df = pd.DataFrame([processed_features], columns=features_names)
        
        # 4. Prediksi
        pred = model.predict(df)[0]
        prediction = "Yes" if pred == 1 else "No"
        
        # Kembalikan fitur mentah ke UI agar form tidak berubah jadi angka semua
        features = raw_features 

    return render_template("predict_view.html", features=features, prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    features = {
        'city': data.get('city'),
        'city_development_index': data.get('city_development_index'),
        'gender': data.get('gender'),
        'relevent_experience': data.get('relevent_experience'),
        'enrolled_university': data.get('enrolled_university'),
        'education_level': data.get('education_level'),
        'major_discipline': data.get('major_discipline'),
        'experience': data.get('experience'),
        'company_size': data.get('company_size'),
        'company_type': data.get('company_type'),
        'last_new_job': data.get('last_new_job'),
        'training_hours': data.get('training_hours')
    }
    # Real prediction using model
    model = load_model()
    import pandas as pd
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    prediction = "Yes" if pred == 1 else "No" if pred == 0 else str(pred)
    return jsonify({
        'prediction': prediction,
        'features': features
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)