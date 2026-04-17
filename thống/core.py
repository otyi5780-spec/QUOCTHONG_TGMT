"""
=============================================================
core.py  —  Cấu hình, Database & Face Recognition Engine
=============================================================
Nhóm 10 · Môn Thị Giác Máy Tính · CV-CT4-K17

Module này gộp 3 thành phần chính vào 1 file để đơn giản cấu trúc:
    1. Config  : Đọc biến môi trường, định nghĩa đường dẫn
    2. Database: ORM (SQLAlchemy) + các hàm CRUD
    3. Face Engine: Train embedding & nhận diện khuôn mặt (DeepFace)
=============================================================
"""

from __future__ import annotations

import os

# Tắt log spam TensorFlow / oneDNN / absl trước mọi import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL",    "3")
os.environ.setdefault("GLOG_minloglevel",       "3")

import logging
import pickle
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import (
    Column, Date, DateTime, Float, ForeignKey,
    Integer, String, create_engine, event, func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*",
)


# =============================================================
#  PHẦN 1: CẤU HÌNH HỆ THỐNG
# =============================================================

load_dotenv()  # Đọc file .env

# ── Đường dẫn thư mục ──
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
PHOTOS_DIR = DATA_DIR / "photos"       # Ảnh thu thập: photos/{mssv}/frame_*.jpg
ATT_DIR    = DATA_DIR / "attendance"   # File CSV điểm danh
CACHE_FILE = DATA_DIR / "embeddings.pkl"  # Cache embeddings đã train
DB_PATH    = DATA_DIR / "db.sqlite3"  # SQLite database

# ── Cấu hình từ .env (hoặc giá trị mặc định) ──
CLASS_NAME  = os.getenv("CLASS_NAME",         "CV-CT4-K17")
FACE_MODEL  = os.getenv("RECOGNITION_MODEL",  "VGG-Face")   # Model DeepFace
THRESHOLD   = float(os.getenv("THRESHOLD",    "0.55"))       # Ngưỡng xác nhận khuôn mặt
CAM_INDEX   = int(os.getenv("CAMERA_INDEX",   "0"))          # Index camera (0 = cam mặc định)
FLASK_PORT  = int(os.getenv("FLASK_PORT",     "5000"))       # Cổng web server

# Tạo các thư mục nếu chưa tồn tại
for directory in (PHOTOS_DIR, ATT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("attendance")


# =============================================================
#  PHẦN 2: DATABASE (SQLAlchemy ORM)
# =============================================================

engine  = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base    = declarative_base()


@event.listens_for(engine, "connect")
def _enable_wal_mode(conn, _):
    """Bật WAL mode cho SQLite — tăng hiệu năng đọc/ghi đồng thời."""
    conn.execute("PRAGMA journal_mode=WAL")


# ── Model: Sinh viên ──
class Student(Base):
    """Bảng lưu thông tin sinh viên."""

    __tablename__ = "students"

    id      = Column(String(20),  primary_key=True)   # MSSV
    name    = Column(String(100), nullable=False)      # Họ tên
    cls     = Column(String(50),  default=CLASS_NAME)  # Lớp
    records = relationship("Record", back_populates="student", cascade="all, delete-orphan")


# ── Model: Bản ghi điểm danh ──
class Record(Base):
    """Bảng lưu lịch sử điểm danh."""

    __tablename__ = "records"

    id         = Column(Integer,    primary_key=True, autoincrement=True)
    student_id = Column(String(20), ForeignKey("students.id"), nullable=False)
    session    = Column(String(50), nullable=False)         # ID buổi học (YYYYMMDD_HHMM)
    date       = Column(Date,       default=date.today)     # Ngày điểm danh
    time       = Column(DateTime,   default=datetime.now)   # Giờ điểm danh
    confidence = Column(Float,      default=0.0)            # Độ tin cậy (0.0 – 1.0)
    method     = Column(String(10), default="face")         # Phương thức: "face" | "manual"
    student    = relationship("Student", back_populates="records")


# Tạo bảng nếu chưa tồn tại
Base.metadata.create_all(engine)


# ─────────────────────────────────────────────────────────────
#  Các hàm CRUD
# ─────────────────────────────────────────────────────────────

def sync_students_from_photos() -> int:
    """
    Đồng bộ thư mục data/photos/{mssv} vào bảng students.

    Hỗ trợ trường hợp người dùng đã chép sẵn dataset lên ổ đĩa nhưng
    database chưa có bản ghi tương ứng, nhờ đó web vẫn hiển thị được
    danh sách sinh viên mà không cần thao tác thêm tay.

    Returns:
        int: Số sinh viên được thêm/cập nhật từ thư mục ảnh.
    """
    synced = 0

    if not PHOTOS_DIR.exists():
        return synced

    with Session() as db:
        for folder in sorted(PHOTOS_DIR.iterdir()):
            if not folder.is_dir():
                continue

            mssv = folder.name.strip()
            if not mssv:
                continue

            info_file = folder / "info.txt"
            full_name = mssv
            cls = CLASS_NAME

            if info_file.exists():
                try:
                    lines = [line.strip() for line in info_file.read_text("utf-8").splitlines()]
                    if lines and lines[0]:
                        full_name = lines[0]
                    if len(lines) > 1 and lines[1]:
                        cls = lines[1]
                except Exception as e:
                    log.warning(f"Không đọc được {info_file}: {e}")

            student = db.get(Student, mssv)
            if student is None:
                db.add(Student(id=mssv, name=full_name, cls=cls))
                synced += 1
                continue

            changed = False
            if full_name and student.name != full_name:
                student.name = full_name
                changed = True
            if cls and student.cls != cls:
                student.cls = cls
                changed = True
            if changed:
                synced += 1

        db.commit()

    if synced:
        log.info(f"Đồng bộ dataset -> database: {synced} sinh viên")
    return synced

def db_add_student(mssv: str, name: str, cls: str = CLASS_NAME):
    """
    Thêm sinh viên mới hoặc cập nhật thông tin nếu MSSV đã tồn tại.

    Args:
        mssv : Mã số sinh viên
        name : Họ tên
        cls  : Tên lớp
    """
    with Session() as db:
        student = db.get(Student, mssv)
        if student:
            student.name = name
            student.cls  = cls
        else:
            db.add(Student(id=mssv, name=name, cls=cls))
        db.commit()


def db_record(mssv: str, session_id: str, confidence: float, method: str = "face") -> bool:
    """
    Ghi nhận điểm danh. Bỏ qua nếu đã điểm danh trong buổi này.

    Args:
        mssv       : Mã số sinh viên
        session_id : ID buổi học
        confidence : Độ tin cậy nhận diện (0.0 – 1.0)
        method     : "face" (camera) hoặc "manual" (tay)

    Returns:
        True nếu ghi thành công, False nếu đã tồn tại.
    """
    with Session() as db:
        duplicate = db.query(Record).filter_by(student_id=mssv, session=session_id).first()
        if duplicate:
            return False
        db.add(Record(student_id=mssv, session=session_id, confidence=confidence, method=method))
        db.commit()
        return True


def db_today() -> list[dict]:
    """
    Lấy tất cả bản ghi điểm danh trong ngày hôm nay.

    Returns:
        list[dict]: Danh sách { id, name, time, conf, session, method }
    """
    with Session() as db:
        rows = db.query(Record).filter(Record.date == date.today()).all()
        return [
            {
                "id":      row.student_id,
                "name":   row.student.name if row.student else "",
                "time":   row.time.strftime("%H:%M:%S"),
                "conf":   round(row.confidence * 100, 1),
                "session": row.session,
                "method": row.method,
            }
            for row in rows
        ]


def db_session(session_id: str) -> list[dict]:
    """
    Lấy bản ghi điểm danh của một buổi học cụ thể.

    Args:
        session_id : ID buổi học (YYYYMMDD_HHMM)

    Returns:
        list[dict]: Danh sách { id, name, time, conf }
    """
    with Session() as db:
        rows = db.query(Record).filter_by(session=session_id).all()
        return [
            {
                "id":   row.student_id,
                "name": row.student.name if row.student else "",
                "time": row.time.strftime("%H:%M:%S"),
                "conf": round(row.confidence * 100, 1),
            }
            for row in rows
        ]


def db_students() -> list[dict]:
    """
    Lấy danh sách toàn bộ sinh viên.

    Returns:
        list[dict]: Danh sách { id, name, cls }
    """
    sync_students_from_photos()
    with Session() as db:
        return [
            {"id": s.id, "name": s.name, "cls": s.cls}
            for s in db.query(Student).order_by(Student.id).all()
        ]


def db_stats() -> dict:
    """
    Thống kê tổng quan hệ thống.

    Returns:
        dict: { total_students, today_count, avg_conf, registered_faces, history, method_stats }
    """
    sync_students_from_photos()
    with Session() as db:
        total_students    = db.query(func.count(Student.id)).scalar()
        today_count       = db.query(func.count(Record.id)).filter(Record.date == date.today()).scalar()
        avg_confidence    = db.query(func.avg(Record.confidence)).scalar() or 0.0
        
        # Thống kê điểm danh 7 ngày qua cho biểu đồ
        from datetime import timedelta
        history = []
        for i in range(6, -1, -1):
            target_date = date.today() - timedelta(days=i)
            cnt = db.query(func.count(Record.id)).filter(Record.date == target_date).scalar()
            history.append({
                "date": target_date.strftime("%d/%m"),
                "count": cnt
            })
            
        # Thống kê phương thức
        face_count   = db.query(func.count(Record.id)).filter(Record.method == "face").scalar()
        manual_count = db.query(func.count(Record.id)).filter(Record.method == "manual").scalar()

        return {
            "total_students":   total_students,
            "today_count":      today_count,
            "avg_conf":         round(avg_confidence * 100, 1),
            "registered_faces": len(_emb),
            "history":          history,
            "method_stats": {
                "face": face_count,
                "manual": manual_count
            }
        }


def db_del_student(mssv: str) -> bool:
    """
    Xoá sinh viên, toàn bộ bản ghi điểm danh và folder ảnh tương ứng.

    Args:
        mssv : Mã số sinh viên cần xoá

    Returns:
        True nếu xoá thành công, False nếu không tìm thấy.
    """
    with Session() as db:
        student = db.get(Student, mssv)
        if student:
            # Xoá folder chứa ảnh dataset
            folder = PHOTOS_DIR / mssv
            if folder.exists():
                import shutil
                try: 
                    shutil.rmtree(folder)
                    log.info(f"🗑️ Đã xoá folder ảnh của {mssv}")
                except Exception as e:
                    log.error(f"Lỗi khi xoá folder ảnh {mssv}: {e}")

            db.delete(student)
            db.commit()
            return True
    return False


def db_manual_checkin(mssv: str, session_id: str) -> bool:
    """
    Điểm danh thủ công (method = 'manual', confidence = 1.0).

    Args:
        mssv       : Mã số sinh viên
        session_id : ID buổi học

    Returns:
        True nếu điểm danh thành công.
    """
    return db_record(mssv, session_id, confidence=1.0, method="manual")


def db_attendance_history(filter_date: str = None, filter_session: str = None) -> list[dict]:
    """
    Lấy lịch sử điểm danh, có thể lọc theo ngày hoặc buổi.

    Args:
        filter_date    : chuỗi 'YYYY-MM-DD' (tuỳ chọn)
        filter_session : chuỗi 'YYYYMMDD_HHMM' (tuỳ chọn)

    Returns:
        list[dict]: Danh sách { id, name, date, time, conf, session, method }
    """
    from datetime import date as date_type
    with Session() as db:
        query = db.query(Record)
        if filter_date:
            try:
                d = date_type.fromisoformat(filter_date)
                query = query.filter(Record.date == d)
            except ValueError:
                pass
        if filter_session:
            query = query.filter(Record.session == filter_session)
        rows = query.order_by(Record.time.desc()).all()
        return [
            {
                "id":      row.student_id,
                "name":   row.student.name if row.student else "",
                "date":   row.date.strftime("%d/%m/%Y"),
                "time":   row.time.strftime("%H:%M:%S"),
                "conf":   round(row.confidence * 100, 1),
                "session": row.session,
                "method": row.method,
            }
            for row in rows
        ]


def db_sessions() -> list[dict]:
    """
    Lấy danh sách các buổi học duy nhất (distinct sessions).

    Returns:
        list[dict]: [{ session, date, count }] sắp xếp mới nhất trước
    """
    with Session() as db:
        from sqlalchemy import distinct
        rows = (
            db.query(
                Record.session,
                Record.date,
                func.count(Record.id).label("count"),
            )
            .group_by(Record.session, Record.date)
            .order_by(Record.date.desc(), Record.session.desc())
            .all()
        )
        return [
            {
                "session": r.session,
                "date":    r.date.strftime("%d/%m/%Y"),
                "count":   r.count,
            }
            for r in rows
        ]


# =============================================================
#  PHẦN 3: FACE RECOGNITION ENGINE (DeepFace)
# =============================================================

# Cache embeddings trong bộ nhớ (load 1 lần duy nhất khi khởi động)
_emb:  dict[str, np.ndarray] = {}  # { mssv : embedding_vector }
_info: dict[str, str]        = {}  # { mssv : họ_tên }


def _read_image_any_path(path: Path) -> Optional[np.ndarray]:
    """
    Đọc ảnh an toàn trên Windows, kể cả khi đường dẫn có dấu.

    cv2.imread() thường lỗi với Unicode path như thư mục "thống", nên
    dùng np.fromfile() + cv2.imdecode() để tránh mất dữ liệu khi train.
    """
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _save_embeddings():
    """Lưu embeddings cache ra file để dùng lại lần sau."""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"emb": _emb, "info": _info}, f)


def load_embeddings():
    """
    Tải embeddings đã train từ file cache vào bộ nhớ.
    Được gọi tự động khi import module.
    """
    global _emb, _info
    if not CACHE_FILE.exists():
        return
    try:
        cached = pickle.load(open(CACHE_FILE, "rb"))
        _emb  = cached.get("emb",  {})
        _info = cached.get("info", {})
        log.info(f"Load embeddings: {len(_emb)} sinh viên")
    except Exception:
        log.warning("Không tải được file embeddings. Cần train lại.")
        _emb, _info = {}, {}


def train_all() -> int:
    """
    Train embedding cho toàn bộ sinh viên trong data/photos/.

    Quy trình mỗi sinh viên:
        1. Đọc tất cả ảnh .jpg trong thư mục {mssv}/
        2. Trích xuất vector đặc trưng bằng DeepFace
        3. Lấy trung bình các vector → chuẩn hoá L2
        4. Lưu vào _emb và cập nhật database

    Returns:
        Số lượng sinh viên được train thành công.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Tắt log TensorFlow
    from deepface import DeepFace

    count = 0

    def collect_images(folder: Path) -> list[Path]:
        image_paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            image_paths.extend(folder.glob(pattern))
            image_paths.extend(folder.glob(pattern.upper()))
        return sorted({p.resolve(): p for p in image_paths}.values(), key=lambda p: p.name)

    for folder in sorted(PHOTOS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        mssv      = folder.name
        image_list = collect_images(folder)
        if not image_list:
            log.warning(f"  [{mssv}] Không có ảnh, bỏ qua.")
            continue

        # Đọc tên sinh viên từ info.txt (nếu có)
        info_file = folder / "info.txt"
        full_name = mssv  # Fallback = MSSV
        if info_file.exists():
            try:
                full_name = info_file.read_text("utf-8").strip().splitlines()[0]
            except Exception:
                pass

        cls_name = CLASS_NAME
        if info_file.exists():
            try:
                info_lines = [line.strip() for line in info_file.read_text("utf-8").splitlines()]
                if info_lines and info_lines[0]:
                    full_name = info_lines[0]
                if len(info_lines) > 1 and info_lines[1]:
                    cls_name = info_lines[1]
            except Exception:
                pass

        # Trích xuất vector đặc trưng từ từng ảnh
        embedding_vectors = []
        for img_path in image_list:
            img = _read_image_any_path(img_path)
            if img is None:
                log.warning(f"  [{mssv}] Không đọc được ảnh: {img_path.name}")
                continue
            try:
                result = DeepFace.represent(
                    img,
                    model_name       = FACE_MODEL,
                    enforce_detection= False,
                    detector_backend = "skip",
                )
                if result:
                    vec = np.array(result[0]["embedding"], dtype=np.float32)
                    embedding_vectors.append(vec)
            except Exception as e:
                log.debug(f"  [{mssv}] Bỏ qua {img_path.name}: {e}")

        if not embedding_vectors:
            log.warning(f"  [{mssv}] Không trích xuất được embedding nào.")
            continue

        # Tính embedding trung bình → chuẩn hoá L2
        avg_embedding = np.mean(embedding_vectors, axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding)

        _emb[mssv]  = avg_embedding
        _info[mssv] = full_name

        # Đồng bộ vào database
        db_add_student(mssv, full_name, cls_name)
        log.info(f"  ✓ Train [{mssv}] {full_name} — {len(embedding_vectors)} ảnh")
        count += 1

    _save_embeddings()
    log.info(f"Train hoàn tất: {count} sinh viên")
    return count


# =============================================================
#  PHẦN 4: ANTI-SPOOFING — Chống giả mạo khuôn mặt
# =============================================================

def is_real_face(face_roi: np.ndarray, threshold: float = 0.45) -> bool:
    """
    Kiểm tra khuôn mặt có phải thật hay không (Anti-Spoofing).

    Tổng hợp 3 kỹ thuật phân tích texture không cần model AI ngoài:

        1. LBP Texture Score  — Da thật có texture phức tạp & không đều.
                                Ảnh in / màn hình thường rất mịn, đồng đều.

        2. FFT Frequency Score — Màn hình / máy chiếu tạo ra Moiré pattern
                                 có thể phát hiện qua phân tích tần số FFT.

        3. Gradient Score      — Mặt thật có gradient cạnh tự nhiên đa dạng.
                                 Ảnh in phẳng thường cho gradient yếu hơn.

    Lưu ý:
        - Không cần cài thêm thư viện, chỉ dùng numpy + opencv sẵn có.
        - Phù hợp làm lớp lọc ban đầu cho hệ thống điểm danh offline.

    Args:
        face_roi  : Ảnh khuôn mặt đã crop (BGR np.ndarray, bất kỳ kích thước)
        threshold : Ngưỡng xác nhận thật (0.0 – 1.0). Mặc định = 0.45.

    Returns:
        True  → Mặt thật    → cho phép tiếp tục nhận diện
        False → Nghi giả mạo → chặn (ảnh in, chiếu màn hình, deepfake đơn giản)
    """
    if face_roi is None or face_roi.size == 0:
        return False

    # ── Resize về kích thước chuẩn để so sánh nhất quán ──
    roi = cv2.resize(face_roi, (128, 128))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    scores = []

    # ─────────────────── 1. LBP Texture Score ───────────────────
    # Tính Local Binary Pattern đơn giản bằng so sánh pixel trung tâm vs 8 lân cận
    # Mặt thật: histogram phân phối đều → std cao
    # Ảnh fake: texture đơn điệu → std thấp
    def lbp_simple(img):
        h, w = img.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = img[1:-1, 1:-1]
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                neighbor = img[1+dy:h-1+dy, 1+dx:w-1+dx]
                lbp += (neighbor >= center).astype(np.uint8)
        return lbp

    lbp = lbp_simple(gray)
    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 8))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    lbp_score = float(np.std(hist))          # Da thật > 0.04, ảnh fake < 0.02
    scores.append(min(lbp_score / 0.08, 1.0))

    # ─────────────────── 2. FFT Frequency Score ───────────────────
    # Ảnh từ màn hình/điện thoại có peak tần số cao (Moiré pattern)
    # Mặt thật: năng lượng tần số cao phân tán đều
    # Màn hình : có spike tần số cụ thể nổi bật
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))
    h, w = magnitude.shape
    center_region = magnitude[h//4:3*h//4, w//4:3*w//4]
    outer_region  = magnitude.copy()
    outer_region[h//4:3*h//4, w//4:3*w//4] = 0
    # Tỷ lệ năng lượng trung tâm / ngoại vi — màn hình thường cao hơn
    ratio = center_region.mean() / (outer_region.mean() + 1e-6)
    # Mặt thật ratio ~ 2-4, màn hình ratio > 5
    fft_score = 1.0 - min(max(ratio - 2.0, 0) / 4.0, 1.0)
    scores.append(fft_score)

    # ─────────────────── 3. Gradient (Sobel) Score ───────────────────
    # Mặt 3D thật có gradient phức tạp theo nhiều hướng
    # Ảnh in phẳng / màn hình thường có gradient yếu và đơn điệu
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    # Độ lệch chuẩn gradient — thật cao, fake thấp
    grad_std = float(np.std(gradient_mag))
    grad_score = min(grad_std / 30.0, 1.0)    # chuẩn hoá vào [0, 1]
    scores.append(grad_score)

    # ─────────────────── Kết hợp 3 điểm → quyết định ───────────────────
    # Trung bình có trọng số: LBP 40%, FFT 30%, Gradient 30%
    final_score = 0.40 * scores[0] + 0.30 * scores[1] + 0.30 * scores[2]

    log.debug(
        f"[AntiSpoof] LBP={scores[0]:.2f} FFT={scores[1]:.2f} "
        f"Grad={scores[2]:.2f} → final={final_score:.2f} "
        f"({'REAL' if final_score >= threshold else 'FAKE'})"
    )

    return final_score >= threshold


def recognize(frame: np.ndarray) -> list[dict]:
    """
    Nhận diện toàn bộ khuôn mặt trong một frame camera.
    Hỗ trợ đa khuôn mặt (multi-face) đồng thời với xử lý song song.

    Quy trình:
        1. Phát hiện tất cả khuôn mặt bằng SSD backend
        2. Trích xuất embedding song song (ThreadPoolExecutor)
        3. So sánh cosine similarity với embeddings đã train
        4. Gán nhãn nếu similarity >= THRESHOLD

    Args:
        frame : Frame BGR từ OpenCV

    Returns:
        list[dict]: Danh sách kết quả (mỗi phần tử = 1 khuôn mặt):
            { sid, name, conf, bbox:(x,y,w,h), known }
    """
    if not _emb:
        return []

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from deepface import DeepFace
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── Bước 1: Phát hiện tất cả khuôn mặt bằng SSD ──
    try:
        faces = DeepFace.extract_faces(
            frame,
            detector_backend="ssd",
            enforce_detection=False,
            align=False
        )
        detected_faces = []
        for face_obj in faces:
            if face_obj["confidence"] > 0.4:
                f = face_obj["facial_area"]
                detected_faces.append((
                    max(0, f["x"]), max(0, f["y"]),
                    max(0, f["w"]), max(0, f["h"])
                ))
    except Exception:
        detected_faces = []

    if not detected_faces:
        return []

    # ── Bước 2: Trích xuất embedding + nhận diện SONG SONG ──
    def process_face(bbox):
        x, y, w, h = bbox
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            return None
        try:
            # ── Anti-Spoofing: Loại bỏ ảnh in / màn hình trước khi nhận diện ──
            if not is_real_face(face_roi):
                return {
                    "sid":   "?",
                    "name":  "⚠ Fake Face",
                    "conf":  0.0,
                    "bbox":  bbox,
                    "known": False,
                    "spoof": True,          # Flag để hiển thị màu riêng trên UI
                }

            rep = DeepFace.represent(
                face_roi,
                model_name        = FACE_MODEL,
                enforce_detection = False,
                detector_backend  = "skip",
            )
            if not rep:
                return {"sid": "?", "name": "Unknown", "conf": 0.0, "bbox": bbox, "known": False}

            query_vec = np.array(rep[0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(query_vec)
            if norm == 0:
                return None
            query_vec /= norm

            # So sánh cosine similarity với toàn bộ sinh viên
            best_sid  = max(_emb.keys(), key=lambda sid: float(np.dot(query_vec, _emb[sid])))
            best_sim  = float(np.dot(query_vec, _emb[best_sid]))
            is_known  = best_sim >= THRESHOLD

            return {
                "sid":   best_sid if is_known else "?",
                "name":  _info.get(best_sid, "Unknown") if is_known else "Unknown",
                "conf":  round(best_sim, 4),
                "bbox":  bbox,
                "known": bool(is_known),
            }
        except Exception:
            return {"sid": "?", "name": "Unknown", "conf": 0.0, "bbox": bbox, "known": False}

    results = []
    # Dùng tối đa min(face_count, 4) worker để xử lý song song
    max_workers = min(len(detected_faces), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_face, bbox): bbox for bbox in detected_faces}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    return results


# Tải embeddings ngay khi import module
sync_students_from_photos()
load_embeddings()
