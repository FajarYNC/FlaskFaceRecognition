import sys
import os
import cv2
import numpy as np
import threading
import mariadb
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from playsound import playsound
import tflite_runtime.interpreter as tflite
from scipy.spatial import distance as dist

# === KONFIGURASI ===
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
EMBEDDING_MODEL_PATH = "mobilefacenet.tflite"
SOUND_PATH = Path("sound/absen.mp3")
REGISTRATION_SOUND_PATH = Path("sound/registration.mp3")
DB_IMAGE_PATH = Path("db_image")
ABSEN_IMAGE_PATH = Path("absen")
CAMERA_URL = "rtsp://cctv_tengah:piramida123@192.168.100.237:554/stream1"
RECOGNITION_THRESHOLD = 0.5
DETECTION_INTERVAL = 10 # Jalankan deteksi penuh setiap 10 frame

# === VARIABEL GLOBAL & FUNGSI UTILITAS (Tidak berubah) ===
# ... (Salin semua dari utakatik_13.py)
known_ids, known_names, known_departments, known_encodings, known_image_paths = [], [], [], [], []
last_logged_time = {}
def load_detector():
    if not Path(YUNET_MODEL_PATH).exists(): return None
    try: return cv2.FaceDetectorYN.create(model=YUNET_MODEL_PATH, config="", input_size=(320, 320), score_threshold=0.6, nms_threshold=0.3, top_k=5000)
    except Exception as e: print(f"Error loading YuNet: {e}"); return None
def load_embedder(): return tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)
def detect_faces_yunet(detector, frame):
    if detector is None: return []
    h, w, _ = frame.shape; detector.setInputSize((w, h)); _, faces = detector.detect(frame); results = []
    if faces is not None:
        for face in faces:
            box = list(map(int, face[:4])); x1, y1, width, height = box; x2, y2 = x1 + width, y1 + height
            results.append((y1, x1, y2, x2))
    return results
def extract_embedding(embedder, face_img):
    emb_input = embedder.get_input_details(); emb_output = embedder.get_output_details(); face_resized = cv2.resize(face_img, (112, 112))
    face_normalized = (face_resized / 255.0 - 0.5) / 0.5; input_tensor = np.expand_dims(face_normalized.astype(np.float32), axis=0)
    embedder.set_tensor(emb_input[0]['index'], input_tensor); embedder.invoke(); embedding = embedder.get_tensor(emb_output[0]['index'])[0]
    return embedding / np.linalg.norm(embedding)
def cosine_similarity(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def connect_db():
    try: return mariadb.connect(user="raspberry", password="pisang123", host="192.168.100.240", port=3306, database="attedance", connect_timeout=5)
    except mariadb.Error as e: print(f"DB Error: {e}"); return None
def load_known_faces():
    global known_ids, known_names, known_departments, known_encodings, known_image_paths
    known_ids.clear(); known_names.clear(); known_departments.clear(); known_encodings.clear(); known_image_paths.clear()
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, department, face_encoding, face_image FROM Users")
            for user_id, name, dept, enc, image_path in cursor.fetchall():
                known_ids.append(user_id); known_names.append(name); known_departments.append(dept)
                known_encodings.append(np.frombuffer(enc, dtype=np.float32)); known_image_paths.append(image_path)
        except mariadb.Error as e: print(f"Error loading faces from DB: {e}")
        finally: conn.close()
    print(f"Loaded {len(known_names)} unique users.")
def record_attendance(user_id):
    conn = connect_db()
    if not conn: return
    try:
        cursor = conn.cursor(); now = datetime.now(); tgl, tipe = now.date(), "masuk" if now.hour < 12 else "pulang"
        cursor.execute("SELECT COUNT(*) FROM AttendanceLog WHERE user_id=%s AND DATE(time)=%s AND type=%s", (user_id, tgl, tipe))
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO AttendanceLog (user_id, time, type) VALUES (%s, %s, %s)", (user_id, now, tipe)); conn.commit()
            if SOUND_PATH.exists(): threading.Thread(target=playsound, args=(str(SOUND_PATH),), daemon=True).start()
    except mariadb.Error as e: print(f"Error recording attendance: {e}")
    finally: conn.close()

# === WORKER THREAD (DIROMBAK TOTAL UNTUK TRACKING) ===
class ProcessingThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, object, str, str, float)
    attendance_logged = pyqtSignal(dict)
    stats_updated = pyqtSignal(str, int)
    status_updated = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.active_trackers = {} # Menyimpan tracker aktif
        self.next_face_id = 0
        self.frame_count = 0

    def run(self):
        self.running = True
        detector = load_detector()
        embedder = load_embedder()
        embedder.allocate_tensors()
        
        if detector is None:
            self.status_updated.emit("CRITICAL: Gagal memuat model detektor YuNet!"); return

        cap = cv2.VideoCapture(CAMERA_URL)
        last_frame_time = time.time()
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue
            
            frame_for_display = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
            
            h, w, _ = frame_for_display.shape
            roi_x1, roi_y1 = w // 4, h // 4; roi_x2, roi_y2 = w - roi_x1, h - roi_y1
            cv2.rectangle(frame_for_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.putText(frame_for_display, "Area Absen", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- LOGIKA DETEKSI & TRACKING ---
            
            # Hanya jalankan deteksi penuh setiap DETECTION_INTERVAL frame
            if self.frame_count % DETECTION_INTERVAL == 0:
                face_locations = detect_faces_yunet(detector, frame_for_display)
                self.update_trackers(face_locations)
            else:
                self.track_faces(frame_for_display)

            # Proses dan gambar setiap tracker yang aktif
            name, department, score, ref_path = "", "", 0.0, None
            for face_id, data in list(self.active_trackers.items()):
                top, left, bottom, right = map(int, data['box'])
                
                # Cek jika tracker berada di dalam ROI untuk diproses
                face_center_x = left + (right - left) // 2
                face_center_y = top + (bottom - top) // 2
                
                if roi_x1 < face_center_x < roi_x2 and roi_y1 < face_center_y < roi_y2:
                    face_crop = rgb_frame[top:bottom, left:right]
                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        try:
                            embedding = extract_embedding(embedder, face_crop)
                            sims = [cosine_similarity(embedding, e) for e in known_encodings]
                            if sims:
                                best_idx = np.argmax(sims)
                                score = sims[best_idx]
                                if score >= RECOGNITION_THRESHOLD:
                                    user_id, name, department, ref_path = (known_ids[best_idx], known_names[best_idx], known_departments[best_idx], known_image_paths[best_idx])
                                    data['name'] = name # Simpan nama di data tracker
                                    now_time = time.time()
                                    if user_id not in last_logged_time or now_time - last_logged_time[user_id] > 10:
                                        last_logged_time[user_id] = now_time; record_attendance(user_id)
                                        now_dt = datetime.now(); snapshot_filename = f"{name.replace(' ', '_')}_{now_dt.strftime('%Y%m%d_%H%M%S')}.jpg"
                                        ABSEN_IMAGE_PATH.mkdir(exist_ok=True); snapshot_path = str(ABSEN_IMAGE_PATH / snapshot_filename); cv2.imwrite(snapshot_path, frame)
                                        log_data = {"name": name, "timestamp": now_dt.strftime('%H:%M:%S'), "department": department, "ref_image_path": ref_path, "snapshot_path": snapshot_path}
                                        self.attendance_logged.emit(log_data); self.status_updated.emit(f"{name} berhasil absen")
                                else:
                                    data['name'] = "Unknown"
                        except Exception as e: print(f"Error processing face ID {face_id}: {e}")
                
                # Gambar bounding box dan nama
                box_color = (0, 255, 0) if data['name'] != "Unknown" else (255, 165, 0)
                cv2.rectangle(frame_for_display, (left, top), (right, bottom), box_color, 2)
                cv2.putText(frame_for_display, f"{data['name']} (ID: {face_id})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            fps = 1 / (time.time() - last_frame_time + 1e-5); last_frame_time = time.time()
            self.stats_updated.emit(f"FPS: {fps:.2f}", len(self.active_trackers))
            self.frame_ready.emit(cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB), ref_path, name, department, score)
            self.frame_count += 1
            
        if cap: cap.release()
    
    def track_faces(self, frame):
        for face_id, data in list(self.active_trackers.items()):
            success, box = data['tracker'].update(frame)
            if success:
                y1, x1, h, w = map(int, box)
                data['box'] = (y1, x1, y1 + h, x1 + w)
                data['disappeared'] = 0
            else:
                data['disappeared'] += 1
                if data['disappeared'] > 5: # Hapus tracker jika hilang selama >5 siklus deteksi
                    del self.active_trackers[face_id]

    def update_trackers(self, face_locations):
        if not face_locations:
            for face_id in list(self.active_trackers.keys()):
                self.active_trackers[face_id]['disappeared'] += 1
                if self.active_trackers[face_id]['disappeared'] > 5:
                    del self.active_trackers[face_id]
            return

        # Ambil titik pusat dari tracker dan deteksi baru
        tracked_centroids = np.array([((data['box'][1] + data['box'][3]) // 2, (data['box'][0] + data['box'][2]) // 2) for data in self.active_trackers.values()])
        detected_centroids = np.array([((left + right) // 2, (top + bottom) // 2) for top, left, bottom, right in face_locations])

        # Cocokkan tracker lama dengan deteksi baru
        if len(tracked_centroids) > 0:
            D = dist.cdist(tracked_centroids, detected_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows, used_cols = set(), set()
            face_ids = list(self.active_trackers.keys())

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row, col] > 50: continue # Jarak maks untuk dianggap sama

                face_id = face_ids[row]
                self.active_trackers[face_id]['box'] = face_locations[col]
                self.active_trackers[face_id]['disappeared'] = 0
                used_rows.add(row); used_cols.add(col)
            
            unused_rows = set(range(tracked_centroids.shape[0])).difference(used_rows)
            for row in unused_rows:
                face_id = face_ids[row]
                self.active_trackers[face_id]['disappeared'] += 1
                if self.active_trackers[face_id]['disappeared'] > 5:
                    del self.active_trackers[face_id]

            unused_cols = set(range(detected_centroids.shape[0])).difference(used_cols)
            for col in unused_cols:
                self.register_new_tracker(face_locations[col])
        else:
            for loc in face_locations:
                self.register_new_tracker(loc)
    
    def register_new_tracker(self, box):
        tracker = cv2.TrackerKCF_create()
        # OpenCV tracker perlu format (x, y, w, h)
        top, left, bottom, right = box
        tracker_box = (left, top, right - left, bottom - top)
        tracker.init(cv2.UMat(np.zeros((1,1), dtype=np.uint8)), tracker_box) # Workaround init
        
        self.active_trackers[self.next_face_id] = {
            'tracker': tracker, 'box': box, 'disappeared': 0, 'name': 'Unknown'
        }
        self.next_face_id += 1

    def stop(self): self.running = False; self.wait()


# === KELAS GUI & REGISTRASI (Tidak berubah dari utakatik_13) ===
class AttendanceApp(QWidget):
    # ... (Salin seluruh kelas AttendanceApp dari utakatik_13.py)
    pass
class RegisterForm(QDialog):
    # ... (Salin seluruh kelas RegisterForm dari utakatik_13.py)
    pass

# Di bawah ini adalah salinan lengkapnya untuk kenyamanan Anda.
class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistem Absensi v14 - Face Tracking")
        self.setGeometry(100, 100, 1000, 750)
        self.processing_thread = None
        self.image_label = QLabel("Kamera Berhenti"); self.image_label.setFixedSize(640, 480); self.image_label.setStyleSheet("border: 1px solid black; background-color: #333;"); self.image_label.setAlignment(Qt.AlignCenter)
        self.live_ref_image_label = QLabel("Live Ref"); self.live_ref_image_label.setFixedSize(150, 150); self.live_ref_image_label.setStyleSheet("border: 1px solid grey;"); self.live_ref_image_label.setAlignment(Qt.AlignCenter)
        self.live_verification_label = QLabel("Menunggu deteksi..."); self.live_verification_label.setWordWrap(True)
        self.log_detail_image_label = QLabel("Log Img"); self.log_detail_image_label.setFixedSize(150, 150); self.log_detail_image_label.setStyleSheet("border: 1px solid grey;"); self.log_detail_image_label.setAlignment(Qt.AlignCenter)
        self.log_detail_info_label = QLabel("Klik log untuk melihat detail."); self.log_detail_info_label.setWordWrap(True)
        self.log_list_widget = QListWidget()
        self.stats_label = QLabel("FPS: 0 | Wajah Dilacak: 0"); self.info_label = QLabel("Status: Siap")
        self.start_btn = QPushButton("Mulai Kamera"); self.stop_btn = QPushButton("Stop Kamera")
        self.register_btn = QPushButton("Registrasi Wajah"); self.export_btn = QPushButton("Export Log ke Excel")
        self.stop_btn.setEnabled(False)
        right_panel = QVBoxLayout()
        live_verification_group = QGroupBox("Verifikasi Real-time"); live_layout = QVBoxLayout(); live_layout.addWidget(self.live_ref_image_label, alignment=Qt.AlignCenter); live_layout.addWidget(self.live_verification_label); live_verification_group.setLayout(live_layout)
        log_detail_group = QGroupBox("Detail Log Terpilih"); log_detail_layout = QVBoxLayout(); log_detail_layout.addWidget(self.log_detail_image_label, alignment=Qt.AlignCenter); log_detail_layout.addWidget(self.log_detail_info_label); log_detail_group.setLayout(log_detail_layout)
        log_list_group = QGroupBox("Log Absensi Terakhir"); log_list_layout = QVBoxLayout(); log_list_layout.addWidget(self.log_list_widget); log_list_group.setLayout(log_list_layout)
        right_panel.addWidget(live_verification_group); right_panel.addWidget(log_detail_group); right_panel.addWidget(log_list_group)
        main_h_layout = QHBoxLayout(); main_h_layout.addWidget(self.image_label); main_h_layout.addLayout(right_panel)
        status_h_layout = QHBoxLayout(); status_h_layout.addWidget(self.stats_label); status_h_layout.addWidget(self.info_label, 1, alignment=Qt.AlignRight)
        button_h_layout = QHBoxLayout(); [button_h_layout.addWidget(btn) for btn in [self.start_btn, self.stop_btn, self.register_btn, self.export_btn]]
        container_layout = QVBoxLayout(); container_layout.addLayout(main_h_layout); container_layout.addLayout(status_h_layout); container_layout.addLayout(button_h_layout)
        self.setLayout(container_layout)
        self.start_btn.clicked.connect(self.start_camera); self.stop_btn.clicked.connect(self.stop_camera)
        self.register_btn.clicked.connect(self.open_register_form); self.export_btn.clicked.connect(self.export_log)
        self.log_list_widget.itemClicked.connect(self.show_log_details)

    def start_camera(self):
        if not self.processing_thread or not self.processing_thread.isRunning():
            self.processing_thread = ProcessingThread(); self.processing_thread.frame_ready.connect(self.update_live_view)
            self.processing_thread.attendance_logged.connect(self.add_log_entry); self.processing_thread.stats_updated.connect(self.update_stats)
            self.processing_thread.status_updated.connect(self.update_status); self.processing_thread.start()
            self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self.register_btn.setEnabled(False)
            self.info_label.setText("Status: Kamera berjalan...")

    def stop_camera(self):
        if self.processing_thread and self.processing_thread.isRunning(): self.processing_thread.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.register_btn.setEnabled(True)
        self.image_label.setText("Kamera Berhenti"); self.info_label.setText("Status: Kamera berhenti")
        self.stats_label.setText("FPS: 0 | Wajah Dilacak: 0"); self.live_ref_image_label.setText("Live Ref")
        self.live_verification_label.setText("Menunggu deteksi..."); self.log_detail_image_label.setText("Log Img"); self.log_detail_info_label.setText("Klik log untuk melihat detail.")

    def update_live_view(self, frame_rgb, ref_path, name, department, score):
        h, w, ch = frame_rgb.shape; qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888); self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        # Panel live view sekarang bisa disederhanakan karena info detail ada di box
        if name and ref_path:
            ref_pixmap = QPixmap(ref_path)
            self.live_ref_image_label.setPixmap(ref_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.live_verification_label.setText(f"<b>Terakhir Cocok:</b> {name}<br><b>Skor:</b> <span style='color:green;'>{score:.2%}</span>")
        else:
            self.live_ref_image_label.setText("Live Ref")
            self.live_verification_label.setText("Menunggu deteksi...")

    def add_log_entry(self, log_data):
        item_text = f"✅ {log_data['timestamp']} - {log_data['name']}"; new_item = QListWidgetItem(item_text); new_item.setData(Qt.UserRole, log_data); self.log_list_widget.insertItem(0, new_item)
        if self.log_list_widget.count() > 100: self.log_list_widget.takeItem(100)

    def show_log_details(self, item):
        log_data = item.data(Qt.UserRole)
        if log_data and Path(log_data['snapshot_path']).exists():
            snapshot_pixmap = QPixmap(log_data['snapshot_path']); self.log_detail_image_label.setPixmap(snapshot_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.log_detail_info_label.setText(f"<b>Nama:</b> {log_data['name']}<br><b>Departemen:</b> {log_data['department']}<br><b>Waktu:</b> <span style='color:blue;'>{log_data['timestamp']}</span>")

    def update_stats(self, fps_str, face_count): self.stats_label.setText(f"{fps_str} | Wajah Dilacak: {face_count}")
    def update_status(self, message): self.info_label.setText(f"Status: {message}")
    def open_register_form(self): self.stop_camera(); reg_form = RegisterForm(self); reg_form.exec_(); load_known_faces()
    def export_log(self):
        conn = connect_db();
        if not conn: QMessageBox.warning(self, "Error", "Koneksi DB gagal"); return
        try:
            query = "SELECT u.name, u.address, u.department, a.time, a.type FROM AttendanceLog a JOIN Users u ON u.id = a.user_id ORDER BY a.time DESC"
            df = pd.read_sql(query, conn); df.columns = ["Nama", "Alamat", "Departemen", "Waktu", "Tipe"]; file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Excel", "", "Excel (*.xlsx)")
            if file_path:
                if not file_path.endswith(".xlsx"): file_path += ".xlsx"
                df.to_excel(file_path, index=False); QMessageBox.information(self, "Sukses", f"Data berhasil diekspor ke {file_path}")
        except Exception as e: QMessageBox.critical(self, "Error", f"Gagal mengekspor data: {e}")
        finally: conn.close()
    def closeEvent(self, event): self.stop_camera(); event.accept()

class RegisterForm(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Registrasi Wajah - Average Embedding"); self.setFixedSize(800, 700)
        self.name_input = QLineEdit(); self.addr_input = QLineEdit()
        self.dept_input = QComboBox(); self.dept_input.addItems(["HRD", "Teknik", "Keuangan", "Lainnya"])
        self.instruksi = QLabel("Ambil 6 foto dari berbagai sudut (mendongak, sedikit kiri/kanan).")
        self.captured_images = [None] * 6; self.captured_encodings = [None] * 6
        self.embedder = load_embedder(); self.embedder.allocate_tensors()
        self.detector = load_detector()
        if self.detector is None: QMessageBox.critical(self, "Error", "Gagal memuat model detektor YuNet."); self.close(); return
        layout = QVBoxLayout(); form_layout = QFormLayout()
        form_layout.addRow("Nama:", self.name_input); form_layout.addRow("Alamat:", self.addr_input); form_layout.addRow("Departemen:", self.dept_input)
        layout.addLayout(form_layout); layout.addWidget(self.instruksi)
        self.video_label = QLabel(); self.video_label.setFixedSize(640, 480); self.video_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter); grid = QGridLayout(); self.thumbnail_labels, self.capture_buttons = [], []
        pose_names = ["Depan", "Senyum", "Kiri", "Kanan", "Atas", "Bawah"]
        for i in range(6):
            btn = QPushButton(f"Ambil {pose_names[i]}"); thumb = QLabel("Belum ada"); thumb.setFixedSize(100, 80); thumb.setStyleSheet("border: 1px solid gray;"); thumb.setAlignment(Qt.AlignCenter)
            self.capture_buttons.append(btn); self.thumbnail_labels.append(thumb); grid.addWidget(btn, i % 3, (i // 3) * 2); grid.addWidget(thumb, i % 3, (i // 3) * 2 + 1)
            btn.clicked.connect(lambda checked, idx=i: self.capture(idx))
        layout.addLayout(grid); self.finish_btn = QPushButton("Selesai Registrasi"); self.finish_btn.setEnabled(False)
        self.finish_btn.clicked.connect(self.save_registration); layout.addWidget(self.finish_btn)
        self.setLayout(layout); self.cam = cv2.VideoCapture(CAMERA_URL); self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame); self.timer.start(30); self.frame = None
    def update_frame(self):
        ret, frame = self.cam.read()
        if ret:
            frame = cv2.resize(frame, (640, 480)); self.frame = frame.copy()
            h, w, _ = frame.shape; x1, y1 = w // 4, h // 4; x2, y2 = w - x1, h - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img))
    def capture(self, index):
        if self.frame is None: return
        face_locations = detect_faces_yunet(self.detector, self.frame)
        if not face_locations: QMessageBox.warning(self, "Error", "Wajah tidak terdeteksi!"); return
        (top, left, bottom, right) = max(face_locations, key=lambda r: (r[2]-r[0])*(r[3]-r[1]))
        face_img = self.frame[top:bottom, left:right]
        try:
            embedding = extract_embedding(self.embedder, face_img); self.captured_images[index] = face_img.copy(); self.captured_encodings[index] = embedding
            thumb_rgb = cv2.cvtColor(cv2.resize(face_img, (100, 80)), cv2.COLOR_BGR2RGB)
            qimg = QImage(thumb_rgb.data, 100, 80, QImage.Format_RGB888); self.thumbnail_labels[index].setPixmap(QPixmap.fromImage(qimg))
            if all(e is not None for e in self.captured_encodings):
                self.finish_btn.setEnabled(True); self.instruksi.setText("Semua foto telah diambil. Klik 'Selesai Registrasi'.")
        except Exception as e: QMessageBox.critical(self, "Error", f"Gagal memproses wajah: {e}")
    def save_registration(self):
        name = self.name_input.text().strip(); addr = self.addr_input.text().strip(); dept = self.dept_input.currentText()
        if not name or not addr: QMessageBox.warning(self, "Gagal", "Nama dan Alamat harus diisi!"); return
        avg_embedding = np.mean(self.captured_encodings, axis=0); avg_embedding /= np.linalg.norm(avg_embedding)
        folder = DB_IMAGE_PATH / name.replace(" ", "_"); folder.mkdir(parents=True, exist_ok=True)
        representative_image = self.captured_images[0]; filename = f"{name.replace(' ', '_')}_representative.jpg"; filepath = str(folder / filename)
        cv2.imwrite(filepath, representative_image)
        conn = connect_db()
        if conn is None: QMessageBox.warning(self, "Error", "Koneksi ke database gagal"); return
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Users (name, address, department, face_image, face_encoding) VALUES (%s, %s, %s, %s, %s)", (name, addr, dept, filepath, avg_embedding.tobytes()))
            conn.commit();
            if REGISTRATION_SOUND_PATH.exists(): threading.Thread(target=playsound, args=(str(REGISTRATION_SOUND_PATH),), daemon=True).start()
            QMessageBox.information(self, "Sukses", "Registrasi berhasil!"); self.close()
        except mariadb.Error as e: QMessageBox.critical(self, "Database Error", f"Gagal menyimpan data: {e}"); conn.rollback()
        finally: conn.close()
    def closeEvent(self, event):
        self.timer.stop()
        if self.cam.isOpened(): self.cam.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_known_faces()
    win = AttendanceApp()
    win.show()
    sys.exit(app.exec_())