import sys
import os
import cv2
import numpy as np
import threading
import mariadb
import time
import configparser
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify
from scipy.spatial import distance as dist
# Menggunakan TensorFlow TFLite interpreter untuk kompatibilitas Windows
import tensorflow.lite as tflite


# --- 1. Konfigurasi & Inisialisasi ---
config = configparser.ConfigParser()
config.read('config.ini')

db_config = config['database']
cam_config = config['camera']

YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
EMBEDDING_MODEL_PATH = "mobilefacenet.tflite"
RECOGNITION_THRESHOLD = 0.5
DETECTION_INTERVAL = 10 
DB_IMAGE_PATH = Path("db_image")
ABSEN_IMAGE_PATH = Path("absen")

# --- Variabel Global untuk Komunikasi Antar Thread ---
output_frame = None
lock = threading.Lock()
camera = None
camera_url = cam_config.get('default_url', '0')
last_status_data = {
    "stats": "FPS: 0 | Wajah Dilacak: 0",
    "live_verification": {"name": None, "score": 0, "ref_path": None},
    "logs": []
}
known_names, known_encodings, known_image_paths = [], [], []
registration_captures = {} # Untuk menyimpan data registrasi sementara

app = Flask(__name__)
import subprocess

# Variabel untuk proses training
training_log = ""
training_process = None
training_status = "idle"


# --- 2. Fungsi Utilitas (Deteksi, Embedding, DB) ---
def load_detector():
    if not Path(YUNET_MODEL_PATH).exists(): return None
    try:
        return cv2.FaceDetectorYN.create(
            model=YUNET_MODEL_PATH, config="", input_size=(320, 320),
            score_threshold=0.6, nms_threshold=0.3, top_k=5000
        )
    except Exception as e:
        print(f"Error loading YuNet: {e}"); return None

def load_embedder():
    return tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)

def detect_faces_yunet(detector, frame):
    if detector is None: return []
    h, w, _ = frame.shape; detector.setInputSize((w, h)); _, faces = detector.detect(frame); results = []
    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4])); x1, y1, width, height = box; x2, y2 = x1 + width, y1 + height
            results.append((y1, x1, y2, x2))
    return results

def extract_embedding(embedder, face_img):
    emb_input = embedder.get_input_details(); emb_output = embedder.get_output_details()
    face_resized = cv2.resize(face_img, (112, 112))
    face_normalized = (face_resized / 255.0 - 0.5) / 0.5
    input_tensor = np.expand_dims(face_normalized.astype(np.float32), axis=0)
    embedder.set_tensor(emb_input[0]['index'], input_tensor); embedder.invoke()
    embedding = embedder.get_tensor(emb_output[0]['index'])[0]
    return embedding / np.linalg.norm(embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def connect_db():
    try:
        return mariadb.connect(
            user=db_config.get('user'), password=db_config.get('password'),
            host=db_config.get('host'), port=int(db_config.get('port')),
            database=db_config.get('dbname'), connect_timeout=5
        )
    except mariadb.Error as e:
        print(f"DB Error: {e}"); return None

def load_known_faces_from_db():
    global known_names, known_encodings, known_image_paths
    known_names.clear(); known_encodings.clear(); known_image_paths.clear()
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name, face_encoding, face_image FROM Users")
            for name, enc, image_path in cursor.fetchall():
                known_names.append(name)
                known_encodings.append(np.frombuffer(enc, dtype=np.float32))
                known_image_paths.append(image_path)
        except mariadb.Error as e: print(f"Error loading faces from DB: {e}")
        finally: conn.close()
    print(f"Loaded {len(known_names)} unique users from database.")

# --- 3. Thread Utama untuk Pemrosesan Video ---
def video_processing_thread():
    global output_frame, camera, camera_url, last_status_data

    detector = load_detector()
    embedder = load_embedder()
    embedder.allocate_tensors()
    
    if detector is None:
        print("CRITICAL: Gagal memuat model detektor YuNet! Thread berhenti.")
        return

    active_trackers = {}; next_face_id = 0; frame_count = 0
    last_logged_time = {}
    last_frame_time = time.time()
    
    while True:
        if camera is None or not camera.isOpened():
            with lock:
                try:
                    cam_source = int(camera_url) if camera_url.isdigit() else camera_url
                    camera = cv2.VideoCapture(cam_source)
                    if not camera.isOpened():
                        camera = None; time.sleep(2); continue
                except Exception as e:
                    print(f"Gagal membuka kamera: {e}"); camera = None; time.sleep(2); continue
        
        ret, frame = camera.read()
        if not ret:
            print("Frame kosong, mencoba menghubungkan ulang...")
            with lock:
                if camera: camera.release()
                camera = None
            time.sleep(2); continue
            
        frame_for_display = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
        
        if frame_count % DETECTION_INTERVAL == 0:
            face_locations = detect_faces_yunet(detector, frame_for_display)
            next_face_id = update_trackers(frame_for_display, face_locations, active_trackers, next_face_id)
        else:
            track_faces(frame_for_display, active_trackers)

        h, w, _ = frame_for_display.shape
        roi_x1, roi_y1 = w // 4, h // 4; roi_x2, roi_y2 = w - roi_x1, h - roi_y1
        cv2.rectangle(frame_for_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        cv2.putText(frame_for_display, "Area Absen", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        live_name, live_score, live_ref_path = None, 0.0, None
        for face_id, data in list(active_trackers.items()):
            top, left, bottom, right = map(int, data['box'])
            face_center_x = left + (right - left) // 2
            face_center_y = top + (bottom - top) // 2
            
            if roi_x1 < face_center_x < roi_x2 and roi_y1 < face_center_y < roi_y2:
                face_crop = rgb_frame[top:bottom, left:right]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    embedding = extract_embedding(embedder, face_crop)
                    sims = [cosine_similarity(embedding, e) for e in known_encodings]
                    if sims:
                        best_idx = np.argmax(sims)
                        score = sims[best_idx]
                        if score >= RECOGNITION_THRESHOLD:
                            name, ref_path = known_names[best_idx], known_image_paths[best_idx]
                            data['name'] = name
                            live_name, live_score, live_ref_path = name, score, ref_path
                            now_time = time.time()
                            if name not in last_logged_time or now_time - last_logged_time[name] > 10:
                                last_logged_time[name] = now_time
                                with lock:
                                    log_entry = f"✅ {datetime.now().strftime('%H:%M:%S')} - {name}"
                                    last_status_data["logs"].insert(0, log_entry)
                                    if len(last_status_data["logs"]) > 10:
                                        last_status_data["logs"].pop()
                        else:
                            data['name'] = "Unknown"
            
            box_color = (0, 255, 0) if data['name'] != "Unknown" else (255, 165, 0)
            cv2.rectangle(frame_for_display, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame_for_display, f"{data['name']} (ID: {face_id})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        with lock:
            output_frame = frame_for_display.copy()
            last_status_data["stats"] = f"FPS: {1 / (time.time() - last_frame_time + 1e-5):.2f} | Wajah Dilacak: {len(active_trackers)}"
            last_status_data["live_verification"] = {"name": live_name, "score": live_score, "ref_path": live_ref_path}
        
        last_frame_time = time.time()
        frame_count += 1

def track_faces(frame, active_trackers):
    for face_id, data in list(active_trackers.items()):
        success, box = data['tracker'].update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            data['box'] = (y, x, y + h, x + w)
            data['disappeared'] = 0
        else:
            data['disappeared'] += 1
            if data['disappeared'] > 5:
                del active_trackers[face_id]

def update_trackers(frame, face_locations, active_trackers, next_face_id):
    if not face_locations and not active_trackers: return next_face_id
    if not face_locations:
        for face_id in list(active_trackers.keys()):
            active_trackers[face_id]['disappeared'] += 1
            if active_trackers[face_id]['disappeared'] > 5: del active_trackers[face_id]
        return next_face_id

    tracked_centroids = np.array([((data['box'][1] + data['box'][3]) // 2, (data['box'][0] + data['box'][2]) // 2) for data in active_trackers.values()]) if active_trackers else np.array([])
    detected_centroids = np.array([((left + right) // 2, (top + bottom) // 2) for top, left, bottom, right in face_locations])

    if len(tracked_centroids) > 0:
        D = dist.cdist(tracked_centroids, detected_centroids)
        rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set(); face_ids = list(active_trackers.keys())
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols or D[row, col] > 75: continue
            face_id = face_ids[row]; active_trackers[face_id]['box'] = face_locations[col]; active_trackers[face_id]['disappeared'] = 0
            used_rows.add(row); used_cols.add(col)
        unused_rows = set(range(len(tracked_centroids))).difference(used_rows)
        for row in unused_rows:
            face_id = face_ids[row]; active_trackers[face_id]['disappeared'] += 1
            if active_trackers[face_id]['disappeared'] > 5: del active_trackers[face_id]
        unused_cols = set(range(len(detected_centroids))).difference(used_cols)
        for col in unused_cols:
            next_face_id = register_new_tracker(frame, face_locations[col], active_trackers, next_face_id)
    else:
        for loc in face_locations:
            next_face_id = register_new_tracker(frame, loc, active_trackers, next_face_id)
    return next_face_id

def register_new_tracker(frame, box, active_trackers, next_face_id):
    top, left, bottom, right = box; tracker_box = (left, top, right - left, bottom - top)
    try: tracker = cv2.TrackerKCF_create(); tracker.init(frame, tracker_box)
    except Exception:
        try: tracker = cv2.legacy.TrackerKCF_create(); tracker.init(frame, tracker_box)
        except Exception as e: print(f"Gagal init tracker: {e}"); return next_face_id
    active_trackers[next_face_id] = {'tracker': tracker, 'box': box, 'disappeared': 0, 'name': 'Unknown'}
    return next_face_id + 1

# --- 4. Flask Routes (API Endpoints) ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register():
    global registration_captures
    registration_captures.clear()
    return render_template("register.html")

# TAMBAHKAN FUNGSI INI
@app.route("/train")
def train():
    return render_template("train.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate_video_stream():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/status")
def status():
    with lock:
        return jsonify(last_status_data)

@app.route("/set_camera", methods=['POST'])
def set_camera():
    global camera, camera_url
    data = request.get_json(); new_url = data.get('url')
    if new_url:
        with lock:
            camera_url = new_url
            if camera: camera.release()
            camera = None
        return jsonify({"status": "success", "message": f"Kamera diubah ke {new_url}"})
    return jsonify({"status": "error", "message": "URL tidak valid"}), 400

@app.route('/capture_pose/<int:pose_index>', methods=['POST'])
def capture_pose(pose_index):
    global output_frame, registration_captures
    with lock:
        if output_frame is None:
            return jsonify({"status": "error", "message": "Kamera belum siap."}), 500
        frame_to_process = output_frame.copy()

    detector = load_detector()
    embedder = load_embedder(); embedder.allocate_tensors()
    
    face_locations = detect_faces_yunet(detector, frame_to_process)
    if not face_locations:
        return jsonify({"status": "error", "message": "Wajah tidak terdeteksi!"}), 400
    
    (top, left, bottom, right) = max(face_locations, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
    face_crop = frame_to_process[top:bottom, left:right]
    
    embedding = extract_embedding(embedder, cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    
    _, buffer = cv2.imencode('.jpg', cv2.resize(face_crop, (100, 80)))
    thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
    
    registration_captures[pose_index] = {
        "embedding": embedding,
        "image": face_crop
    }
    
    return jsonify({"status": "success", "thumbnail": f"data:image/jpeg;base64,{thumbnail_base64}"})

@app.route('/save_registration', methods=['POST'])
def save_registration():
    global registration_captures
    data = request.get_json()
    name = data.get('name'); addr = data.get('address'); dept = data.get('department')

    if len(registration_captures) < 6:
        return jsonify({"status": "error", "message": "Harap ambil semua 6 foto."}), 400
    if not name or not addr:
        return jsonify({"status": "error", "message": "Nama dan Alamat harus diisi."}), 400

    embeddings = [v['embedding'] for k, v in registration_captures.items()]
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding /= np.linalg.norm(avg_embedding)

    folder = DB_IMAGE_PATH / name.replace(" ", "_"); folder.mkdir(parents=True, exist_ok=True)
    representative_image = registration_captures[0]['image']
    filepath = str(folder / f"{name.replace(' ', '_')}_representative.jpg")
    cv2.imwrite(filepath, representative_image)
    
    conn = connect_db()
    if conn is None:
        return jsonify({"status": "error", "message": "Koneksi database gagal."}), 500
    
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Users (name, address, department, face_image, face_encoding) VALUES (%s, %s, %s, %s, %s)",
                       (name, addr, dept, filepath, avg_embedding.tobytes()))
        conn.commit()
        load_known_faces_from_db()
        return jsonify({"status": "success", "message": "Registrasi berhasil!"})
    except mariadb.Error as e:
        conn.rollback()
        return jsonify({"status": "error", "message": f"Database error: {e}"}), 500
    finally:
        conn.close()

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_log, training_process, training_status

    if training_process is not None and training_process.poll() is None:
        return jsonify({"status": "running", "message": "Proses training masih berjalan."}), 400

    training_log = ""
    training_status = "running"

    def run_training():
        global training_log, training_status
        process = subprocess.Popen(
            [sys.executable, "train_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            training_log += line
        process.wait()
        training_status = "finished"

    training_process_thread = threading.Thread(target=run_training)
    training_process_thread.start()

    return jsonify({"status": "started"})

@app.route('/training_status')
def training_status_api():
    global training_log, training_status
    return jsonify({
        "status": training_status,
        "log": training_log
    })

@app.route('/reload_classifier', methods=['POST'])
def reload_classifier():
    # Jika kamu ingin meload ulang model dari classifier.pkl, tambahkan logic di sini
    return jsonify({"status": "success", "message": "Model berhasil dimuat ulang (simulasi)."})


# --- 5. Main Execution Block ---
if __name__ == '__main__':
    load_known_faces_from_db()
    
    processing_thread = threading.Thread(target=video_processing_thread)
    processing_thread.daemon = True
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
