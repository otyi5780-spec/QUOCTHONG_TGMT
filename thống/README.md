# Hệ Thống Điểm Danh Tự Động Bằng Nhận Diện Khuôn Mặt

> **Lớp:** CV-CT4-K17 · **Môn:** Xử Lý Ảnh Số  
> **Công nghệ:** Python · OpenCV · DeepFace · Flask · SQLite

---

## 📋 Mô Tả Đề Tài

Hệ thống điểm danh sinh viên **tự động theo thời gian thực** qua webcam, sử dụng:
- **OpenCV** — Detect khuôn mặt (Haar Cascade) + xử lý frame
- **DeepFace** — Trích xuất embedding khuôn mặt (VGG-Face model)
- **Cosine Similarity** — So sánh khuôn mặt với database sinh viên
- **Flask** — Web dashboard quản lý
- **SQLite + SQLAlchemy** — Lưu trữ dữ liệu

---

## 🗂️ Cấu Trúc Thư Mục

```
attendance_system/
│
├── main.py                    # Entry point chính (4 chế độ)
│
├── core/
│   ├── config.py              # Cấu hình từ .env
│   ├── database.py            # SQLite ORM (Student, AttendanceRecord)
│   ├── face_engine.py         # DeepFace embedding + nhận diện
│   └── attendance_service.py  # Điều phối toàn bộ luồng
│
├── app/
│   └── routes.py              # Flask REST API (8 endpoints)
│
├── web/
│   ├── templates/index.html   # Dashboard HTML (dark UI)
│   └── static/
│       ├── style.css          # Premium dark CSS
│       └── app.js             # Dashboard JS
│
├── scripts/
│   ├── register_students.py   # Đăng ký hàng loạt từ CSV
│   └── export_attendance.py   # Xuất Excel toàn bộ
│
├── tests/
│   └── test_face_engine.py    # Unit tests
│
├── data/
│   ├── students/              # Ảnh sinh viên (MSSV/face_X.jpg)
│   └── attendance/            # File CSV điểm danh
│
├── .env.example               # Template biến môi trường
└── requirements.txt
```

---

## ⚡ Cài Đặt & Chạy

### 1. Tạo môi trường ảo

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/macOS
```

### 2. Cài thư viện

```bash
pip install -r requirements.txt
```

### 3. Tạo file .env

```bash
copy .env.example .env
# Sửa CLASS_NAME, TEACHER_NAME, SUBJECT theo thực tế
```

### 4. Đăng ký sinh viên

**Cách 1: Chụp qua webcam**
```bash
python main.py --register
# Nhập MSSV, Tên → Nhấn SPACE chụp 3-5 ảnh
```

**Cách 2: Từ file ảnh có sẵn**
```
Cấu trúc thư mục:
  data/students/
    2151001/
      face_0.jpg
      face_1.jpg
      info.txt     ← dòng 1: Họ Tên
    2151002/
      face_0.jpg
      info.txt
```
```bash
python main.py --build-db
```

**Cách 3: Đăng ký loạt từ CSV**
```bash
python scripts/register_students.py --csv students.csv
```

### 5. Chạy điểm danh realtime

```bash
python main.py
```
- `SPACE` / Nhận diện tự động mỗi 5 frame
- `S` — Lưu CSV buổi học
- `N` — Bắt đầu buổi học mới
- `Q` / ESC — Thoát

### 6. Mở Web Dashboard

```bash
python main.py --dashboard
# Truy cập: http://localhost:5000
```

---

## 🌐 REST API (Flask)

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/api/stats` | Thống kê tổng quan |
| `GET`  | `/api/today` | Điểm danh hôm nay |
| `GET`  | `/api/session/<id>` | Điểm danh theo buổi |
| `GET`  | `/api/students` | Danh sách sinh viên |
| `POST` | `/api/students` | Thêm sinh viên |
| `DELETE` | `/api/students/<id>` | Xoá sinh viên |
| `POST` | `/api/checkin` | Điểm danh thủ công |
| `POST` | `/api/new-session` | Tạo buổi học mới |
| `GET`  | `/api/export` | Export CSV |
| `POST` | `/api/rebuild-db` | Rebuild face DB |

---

## 🔬 Luồng Hoạt Động

```
Webcam Frame
    │
    ▼
OpenCV Haar Cascade ──→ Detect bounding box khuôn mặt
    │
    ▼
DeepFace.represent() ──→ Extract embedding vector (VGG-Face)
    │
    ▼
Cosine Similarity ──→ So sánh với embedding cache (SQLite)
    │
    ├── Độ tương đồng ≥ 0.55 → Nhận diện được → Ghi điểm danh DB
    └── Độ tương đồng < 0.55 → "Không xác định"
```

---

## 🧪 Chạy Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Xuất Dữ Liệu

```bash
# Export Excel tất cả
python scripts/export_attendance.py

# Export CSV qua Dashboard
http://localhost:5000 → Tab "Xuất Dữ Liệu"
```

---

## ⚙️ Cấu hình (.env)

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `CAMERA_INDEX` | `0` | Index webcam |
| `RECOGNITION_MODEL` | `VGG-Face` | Model nhận dạng: VGG-Face, Facenet, ArcFace |
| `CONFIDENCE_THRESHOLD` | `0.55` | Ngưỡng chấp nhận (0-1) |
| `DETECTION_BACKEND` | `opencv` | Backend detect: opencv, retinaface, mtcnn |
| `CLASS_NAME` | `CV-CT4-K17` | Tên lớp |
| `FLASK_PORT` | `5000` | Port web dashboard |

---

## 👥 Nhóm Thực Hiện
- Đề tài: **Hệ thống điểm danh tự động bằng nhận diện khuôn mặt**
- Công nghệ: Python, OpenCV, DeepFace, Flask, SQLite
