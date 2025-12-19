"""
Script untuk memverifikasi urutan fitur dan format data
Jalankan script ini untuk memastikan model menerima data dengan urutan yang benar
"""

import pandas as pd
from model_util import set_features, apply_preprocessing, load_model

# 1. Cek urutan fitur yang diharapkan
print("=" * 60)
print("URUTAN FITUR YANG DIHARAPKAN MODEL:")
print("=" * 60)
features = set_features()
for i, feat in enumerate(features, 1):
    print(f"{i:2d}. {feat}")

# 2. Test data sample (sesuai dengan data dari CSV)
print("\n" + "=" * 60)
print("TEST DATA SAMPLE:")
print("=" * 60)

test_input = {
    "city": "103",
    "city_development_index": "0.92",
    "gender": "0",  # Male
    "relevent_experience": "1",  # Has relevant experience
    "enrolled_university": "0",  # No enrollment
    "education_level": "2",  # Graduate
    "major_discipline": "0",  # STEM
    "experience": "21",
    "company_size": "7",  # Unknown -> akan jadi angka tertentu
    "company_type": "5",  # Unknown
    "last_new_job": "1",
    "training_hours": "36"
}

# 3. Proses preprocessing
processed = apply_preprocessing(test_input)

print("Input mentah:")
for k, v in test_input.items():
    print(f"  {k}: {v} (type: {type(v).__name__})")

print("\nSetelah preprocessing:")
for k, v in processed.items():
    print(f"  {k}: {v} (type: {type(v).__name__})")

# 4. Buat DataFrame dengan urutan yang benar
df = pd.DataFrame([processed], columns=features)
print("\n" + "=" * 60)
print("DATAFRAME UNTUK PREDIKSI:")
print("=" * 60)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("\nData:")
print(df)
print("\nData types:")
print(df.dtypes)

# 5. Test prediksi (jika model tersedia)
try:
    model = load_model()
    if model:
        print("\n" + "=" * 60)
        print("TEST PREDIKSI:")
        print("=" * 60)
        prediction = model.predict(df)[0]
        result = "Yes (akan pindah kerja)" if prediction == 1 else "No (akan tetap)"
        print(f"Hasil prediksi: {prediction} -> {result}")
        print("\n✅ Model berhasil melakukan prediksi!")
        print("✅ Urutan fitur SUDAH BENAR!")
except Exception as e:
    print(f"\n❌ Error saat prediksi: {e}")
    print("Periksa apakah model sudah di-load dengan benar")

print("\n" + "=" * 60)
print("VERIFIKASI SELESAI")
print("=" * 60)
