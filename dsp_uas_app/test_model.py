import os
import joblib
from model_util import get_model
from dotenv import load_dotenv

def test_get_model_logic():
    print("=== Memulai Test Get Model ===")
    
    # 1. Load environment variables
    load_dotenv()
    username = os.getenv('DAGSHUB_USERNAME')
    token = os.getenv('DAGSHUB_TOKEN')
    
    if not username or not token:
        print("[FAIL] .env tidak terbaca atau kosong. Pastikan DAGSHUB_USERNAME dan DAGSHUB_TOKEN ada.")
        return

    # 2. Hapus model lokal jika ada (untuk benar-benar ngetes proses download)
    model_path = "model/rf_uas_model.pkl"
    if os.path.exists(model_path):
        print(f"[INFO] Menghapus model lama di {model_path} untuk simulasi download ulang...")
        os.remove(model_path)

    # 3. Panggil fungsi get_model
    try:
        print("[INFO] Menghubungi DagsHub MLflow (ini mungkin memakan waktu)...")
        model = get_model()
        
        # 4. Validasi hasil
        if model is not None:
            print("[SUCCESS] Fungsi get_model() mengembalikan objek model.")
            
            # Cek apakah file benar-benar tersimpan di folder model/
            if os.path.exists(model_path):
                print(f"[SUCCESS] File model ditemukan di: {model_path}")
                
                # Cek apakah file bisa di-load ulang (integritas file)
                loaded_model = joblib.load(model_path)
                print("[SUCCESS] File model valid dan bisa di-load menggunakan joblib.")
            else:
                print("[FAIL] Objek ada, tapi file .pkl tidak tersimpan di folder model/")
        else:
            print("[FAIL] Fungsi get_model() mengembalikan None.")
            
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat eksekusi: {e}")

if __name__ == "__main__":
    test_get_model_logic()