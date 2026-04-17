

# Tắt log spam của TensorFlow / oneDNN trước khi import bất kỳ thứ gì
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"   # Tắt INFO/WARNING của TF C++
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"   # Tắt oneDNN verbose
os.environ["ABSL_MIN_LOG_LEVEL"]      = "3"   # Tắt absl log
os.environ["GLOG_minloglevel"]         = "3"   # Tắt glog

import csv
import io
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("attendance")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*",
)


def save_jpg(path: Path, image, quality: int = 95) -> bool:
    """
    Ghi ảnh JPEG theo cách an toàn hơn trên Windows.

    Tránh phụ thuộc hoàn toàn vào cv2.imwrite(), vốn có thể trả False âm thầm
    trong một số môi trường dù chương trình vẫn tiếp tục đếm ảnh như thành công.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return False

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return False

    path.write_bytes(buf.tobytes())
    return path.exists() and path.stat().st_size > 0


# ─────────────────────────────────────────────────────────────────────────────
# 1. THU THẬP ẢNH + TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def step_collect_and_train():
    """Thu thập ảnh khuôn mặt sinh viên rồi train luôn."""
    mssv = input("  MSSV      : ").strip()
    name = input("  Họ Tên   : ").strip()
    cls  = input("  Lớp (Enter = CV-CT4-K17): ").strip() or "CV-CT4-K17"

    if not mssv or not name:
        print("  ⚠  MSSV và Họ Tên không được để trống!\n")
        return

    # ── Thu thập ảnh ──
    from core import PHOTOS_DIR

    save_dir = PHOTOS_DIR / mssv
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "info.txt").write_text(f"{name}\n{cls}", encoding="utf-8")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    TOTAL        = 150
    count        = 0
    is_capturing = False

    log.info(f"Thu thập ảnh: [{mssv}] {name}  — mục tiêu {TOTAL} ảnh")
    print("  SPACE = bắt đầu chụp    Q / ESC = thoát sớm\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        # Thanh trạng thái
        txt = f"[{count}/{TOTAL}]  {'Capturing...' if is_capturing else 'Space Start'}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (15, 15, 30), -1)
        cv2.putText(frame, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 220, 255) if is_capturing else (200, 200, 200), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (0, 220, 60) if is_capturing else (100, 100, 100), 2)

        # Chụp tự động
        if is_capturing and len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            img_path = save_dir / f"frame_{count:03d}.jpg"

            if save_jpg(img_path, face_img):
                count += 1
            else:
                log.warning(f"Không lưu được ảnh: {img_path}")

            if count >= TOTAL:
                log.info(f"✓ Đã chụp đủ {TOTAL} ảnh")
                break
            time.sleep(0.05)

        cv2.imshow(f"Thu Thập Ảnh — {name}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            is_capturing = True
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    actual_count = len(list(save_dir.glob("*.jpg")))
    log.info(f"Lưu {actual_count} ảnh → {save_dir}")

    if actual_count == 0:
        print("  Không có ảnh nào được chụp. Train bị bỏ qua.\n")
        return

    # ── Train ngay sau khi thu thập xong ──
    print("\n  Đang train embeddings...\n")
    from core import train_all
    n = train_all()
    print(f"\n  ✓ Train hoàn tất: {n} sinh viên trong hệ thống.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEB DASHBOARD (điểm danh + quản lý + xem dataset)
# ─────────────────────────────────────────────────────────────────────────────

def step_dashboard():
    """Khởi động web dashboard — truy cập tại http://localhost:5000"""
    from app import create_app
    import core as C

    app = create_app()
    print(f"\n  Web Dashboard → http://localhost:{C.FLASK_PORT}")
    print("  Nhấn Ctrl+C để dừng server.\n")
    app.run(host="0.0.0.0", port=C.FLASK_PORT, debug=False)


# ─────────────────────────────────────────────────────────────────────────────
# MENU CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

MENU = """
  0.  Thoát
  1.  Thu thập ảnh + Train sinh viên mới
  2.  Mở Web Dashboard (điểm danh + quản lý)
"""


def main():
    print("\n  Hệ Thống Điểm Danh Tự Động ")
    print("  " + "─" * 40)

    while True:
        print(MENU)
        choice = input("  Chọn [0-2]: ").strip()

        if choice == "0":
            print("\n  Tạm biệt!\n")
            sys.exit(0)

        elif choice == "1":
            step_collect_and_train()

        elif choice == "2":
            step_dashboard()

        else:
            print("  ⚠  Nhập 0, 1 hoặc 2.\n")


if __name__ == "__main__":
    main()
