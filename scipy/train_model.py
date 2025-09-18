import mariadb
import pickle
import numpy as np
import configparser
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# --- Konfigurasi ---
CONFIG_PATH = "config.ini"
CLASSIFIER_PATH = "classifier.pkl"

# --- Fungsi Utilitas ---
def connect_db(db_config):
    """Menghubungkan ke database MariaDB."""
    try:
        return mariadb.connect(
            user=db_config.get('user'), password=db_config.get('password'),
            host=db_config.get('host'), port=int(db_config.get('port')),
            database=db_config.get('dbname'), connect_timeout=5
        )
    except mariadb.Error as e:
        print(f"DB Error: {e}")
        return None

def main():
    """
    Fungsi utama untuk melatih dan menyimpan classifier.
    """
    print("--- Memulai Proses Training Classifier ---")

    # 1. Baca konfigurasi database
    print("1. Membaca file konfigurasi...")
    config = configparser.ConfigParser()
    if not Path(CONFIG_PATH).exists():
        print(f"ERROR: File '{CONFIG_PATH}' tidak ditemukan!")
        sys.exit(1)
    config.read(CONFIG_PATH)
    db_config = config['database']

    # 2. Ambil data dari database
    print("2. Mengambil data embedding dari database...")
    conn = connect_db(db_config)
    if not conn:
        print("ERROR: Gagal terhubung ke database. Proses dibatalkan.")
        sys.exit(1)
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name, face_encoding FROM Users")
        data = cursor.fetchall()
    except mariadb.Error as e:
        print(f"ERROR: Gagal mengambil data dari tabel Users: {e}")
        sys.exit(1)
    finally:
        conn.close()

    if len(data) == 0:
        print("WARNING: Tidak ada data di database untuk dilatih.")
        sys.exit(0)
    
    unique_names = len(set([name for name, enc in data]))
    if unique_names < 2:
        print(f"ERROR: Butuh setidaknya 2 orang yang berbeda untuk dilatih. Ditemukan: {unique_names}")
        sys.exit(1)

    known_embeddings = [np.frombuffer(enc, dtype=np.float32) for name, enc in data]
    labels = [name for name, enc in data]
    print(f"   -> Ditemukan {len(labels)} total data embedding dari {unique_names} orang.")

    # 3. Latih classifier
    print("3. Menyiapkan label dan melatih SVM Classifier...")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(known_embeddings, labels_encoded)
    print("   -> Training selesai.")

    # 4. Simpan classifier ke file
    print(f"4. Menyimpan classifier ke file: {CLASSIFIER_PATH}")
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump({"recognizer": recognizer, "le": le}, f)

    print("\n--- Proses Training Berhasil! ---")
    print("Classifier baru telah disimpan. Harap restart aplikasi web untuk menggunakan model baru.")
    sys.exit(0)

if __name__ == "__main__":
    main()
