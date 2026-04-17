"""
app.py — Flask Web Dashboard
Nhóm 10 · Môn Thị Giác Máy Tính · CV-CT4-K17

Routes:
    GET  /                        Trang chủ
    GET  /api/stream              MJPEG camera stream (cv2.cap.read → browser)
    POST /api/stream/start        Bắt đầu stream + nhận diện
    POST /api/stream/stop         Dừng stream
    GET  /api/stream/status       Trạng thái camera
    GET  /api/stats               Thống kê tổng quan
    GET  /api/today               Điểm danh hôm nay
    GET  /api/attendance          Lịch sử điểm danh
    GET  /api/sessions            Danh sách buổi học
    GET  /api/students            Danh sách sinh viên
    GET  /api/dataset             Dataset sinh viên
    POST /api/students            Thêm sinh viên
    DEL  /api/students/<mssv>     Xoá sinh viên
    POST /api/train               Train lại embeddings
    GET  /api/export              Xuất CSV
"""

import csv
import io
import logging
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_cors import CORS

log = logging.getLogger("attendance.app")


# ═══════════════════════════════════════════════════════════════
#  Camera State — dùng chung giữa các requests
# ═══════════════════════════════════════════════════════════════

cam_state = {
    "running":    False,       # Đang stream không
    "loading":    False,       # Đang warmup/khởi động
    "load_step":  "",          # Bước hiện tại đang load
    "load_pct":   0,           # Phần trăm hoàn thành (0-100)
    "cap":        None,        # cv2.VideoCapture
    "session":    "",          # ID buổi học
    "frame":      None,        # Frame mới nhất (bytes JPEG)
    "frame_lock": threading.Lock(),
    "frame_counts": {},        # { mssv: { count, name, conf } }
    "confirmed":  {},          # { mssv: { name, conf, time } }
    "results":    [],          # Kết quả nhận diện frame hiện tại
    "thread":     None,        # Thread chạy camera loop
}


def _camera_loop(C):
    """
    Thread chạy nền: đọc frame từ camera, chạy face recognition,
    vẽ bounding box, encode JPEG, lưu vào cam_state["frame"].
    """
    cap = cam_state["cap"]
    CONFIRM_SECONDS    = 5.0   # Giữ mặt liên tục 5 giây để xác nhận điểm danh
    LOST_GRACE_SECONDS = 0.8   # Dung sai ngắn khi model hụt 1-2 nhịp nhận diện

    fc = 0  # frame counter

    while cam_state["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        fc += 1
        disp = frame.copy()

        # ── Nhận diện mỗi 5 frame để tối ưu hiệu năng ──
        if fc % 5 == 0:
            results = C.recognize(frame)
            cam_state["results"] = results
            now = time.monotonic()

            session_id = cam_state["session"]
            fc_map     = cam_state["frame_counts"]
            confirmed  = cam_state["confirmed"]

            # Cập nhật khuôn mặt đang hiện diện, xoá tiến trình nếu mất mặt quá lâu
            current_sids = {r["sid"] for r in results if r["known"]}
            for sid in list(fc_map):
                if sid in confirmed:
                    continue
                if sid not in current_sids and now - fc_map[sid]["last_seen"] > LOST_GRACE_SECONDS:
                    fc_map.pop(sid, None)

            for r in results:
                if not r["known"]:
                    continue
                sid = r["sid"]
                if sid not in fc_map:
                    fc_map[sid] = {
                        "name":       r["name"],
                        "conf":       r["conf"],
                        "started_at": now,
                        "last_seen":  now,
                    }
                else:
                    fc_map[sid]["last_seen"] = now
                    fc_map[sid]["name"]      = r["name"]
                    fc_map[sid]["conf"]      = r["conf"]

                elapsed = now - fc_map[sid]["started_at"]

                # Giữ mặt đủ 5 giây → ghi điểm danh
                if elapsed >= CONFIRM_SECONDS and sid not in confirmed:
                    ok_rec = C.db_record(sid, session_id, r["conf"], method="face")
                    if ok_rec or True:   # luôn add vào confirmed để hiện UI
                        # Crop khuôn mặt cuối cùng
                        x, y, w, h = r["bbox"]
                        face_px = frame[max(0,y):max(0,y+h), max(0,x):max(0,x+w)]
                        b64_img = ""
                        if face_px.size > 0:
                            import base64
                            _, buf = cv2.imencode(".jpg", cv2.resize(face_px, (100, 100)), [cv2.IMWRITE_JPEG_QUALITY, 70])
                            b64_img = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

                        confirmed[sid] = {
                            "name":    r["name"],
                            "session": session_id,
                            "time":    datetime.now().strftime("%H:%M:%S"),
                            "date":    datetime.now().strftime("%d/%m/%Y"),
                            "avatar":  b64_img,
                        }
                        log.info(f"✓ Điểm danh: {r['name']} ({sid}) conf={r['conf']:.2f}")

        # ── Vẽ bounding box lên disp ──
        results = cam_state["results"]
        fc_map  = cam_state["frame_counts"]
        confirmed = cam_state["confirmed"]

        for r in results:
            x, y, w, h = r["bbox"]
            sid          = r["sid"]
            is_confirmed = sid in confirmed
            is_spoof     = r.get("spoof", False)

            # Màu box
            if is_spoof:
                color = (0, 100, 255)       # cam đỏ — fake face bị chặn
            elif is_confirmed:
                color = (34, 197, 94)       # xanh lá — đã xác nhận
            elif r["known"]:
                color = (250, 204, 21)      # vàng — đang đếm
            else:
                color = (60, 60, 220)       # đỏ — không nhận ra

            cv2.rectangle(disp, (x, y), (x+w, y+h), color, 2)

            # Nhãn
            if is_spoof:
                label = "FAKE — Bi chong!"
            elif r["known"]:
                if not is_confirmed and sid in fc_map:
                    elapsed = min(time.monotonic() - fc_map[sid]["started_at"], CONFIRM_SECONDS)
                    remain = max(0.0, CONFIRM_SECONDS - elapsed)
                    label = f"{r['name']}  {r['conf']*100:.0f}%  {remain:.1f}s"
                else:
                    label = f"{'[OK] ' if is_confirmed else ''}{r['name']}  {r['conf']*100:.0f}%"
            else:
                label = "Unknown"

            lbl_bg = (0, 0, 0)
            cv2.rectangle(disp, (x, y - 26), (x + len(label)*9 + 10, y), lbl_bg, -1)
            cv2.putText(disp, label, (x + 5, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Thanh tiến trình (nếu đang đếm)
            if r["known"] and not is_confirmed and sid in fc_map:
                pct = min((time.monotonic() - fc_map[sid]["started_at"]) / CONFIRM_SECONDS, 1.0)
                bar_w = int(w * pct)
                cv2.rectangle(disp, (x, y + h + 2), (x + w, y + h + 7), (40, 40, 40), -1)
                cv2.rectangle(disp, (x, y + h + 2), (x + bar_w, y + h + 7), (250, 204, 21), -1)

            # Overlay mờ khi confirmed
            if is_confirmed:
                overlay = disp.copy()
                cv2.rectangle(overlay, (x+1, y+1), (x+w-1, y+h-1), (34, 197, 94), -1)
                cv2.addWeighted(overlay, 0.15, disp, 0.85, 0, disp)

        # ── Header info ──
        h_frame, w_frame = disp.shape[:2]
        session_lbl = f"Buoi: {cam_state['session']}  |  Da diem: {len(confirmed)}"
        cv2.rectangle(disp, (0, 0), (w_frame, 36), (15, 15, 30), -1)
        cv2.putText(disp, session_lbl, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

        # ── Encode JPEG ──
        _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with cam_state["frame_lock"]:
            cam_state["frame"] = buf.tobytes()

    cap.release()
    cam_state["cap"] = None
    log.info("Camera loop dừng.")


def _mjpeg_generator():
    """Generator cho MJPEG stream — yield từng frame boundary."""
    while cam_state["running"]:
        with cam_state["frame_lock"]:
            frame = cam_state["frame"]
        if frame is None:
            time.sleep(0.033)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )
        time.sleep(0.033)   # ~30fps cap


# ═══════════════════════════════════════════════════════════════
#  Flask App Factory
# ═══════════════════════════════════════════════════════════════

def create_app() -> Flask:
    app = Flask(__name__, static_folder="web/static", template_folder="web/templates")
    CORS(app)
    app.logger.disabled = True
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    import core as C

    # ── Trang chủ ──
    @app.get("/")
    def index():
        return render_template("index.html")

    # ─────────────────────────────────────────────────────────
    #  CAMERA STREAM (MJPEG)
    # ─────────────────────────────────────────────────────────

    @app.get("/api/stream")
    def video_stream():
        """MJPEG stream — nhúng trực tiếp vào <img src='/api/stream'>"""
        if not cam_state["running"]:
            # Trả ảnh placeholder khi camera chưa bật
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera chua bat", (160, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
            cv2.putText(placeholder, "Nhan 'Bat Camera' de bat dau", (90, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
            _, buf = cv2.imencode(".jpg", placeholder)
            return Response(buf.tobytes(), mimetype="image/jpeg")

        return Response(
            _mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/stream/start")
    def stream_start():
        """Khởi động camera: warmup model → load dataset → mở cam."""
        if cam_state["running"] or cam_state["loading"]:
            return jsonify(ok=True, msg="Đang chạy rồi")

        body       = request.json or {}
        session_id = body.get("session", datetime.now().strftime("%Y%m%d_%H%M"))

        if not C._emb:
            students = C.db_students()
            jpg_count = sum(len(list((C.PHOTOS_DIR / s["id"]).glob("*.jpg"))) for s in students)
            if not students:
                msg = "Chưa có sinh viên nào trong dataset."
            elif jpg_count == 0:
                msg = "Đã có sinh viên nhưng chưa có ảnh .jpg để train."
            else:
                msg = "Chưa train embeddings. Nhấn 'Train Lại' hoặc chạy option 1 trước."
            return jsonify(ok=False, error=msg), 400

        def _start_async():
            """Chạy nền: warmup → load dataset → mở camera."""
            try:
                cam_state["loading"]   = True
                cam_state["load_pct"]  = 0

                # ── Bước 1: Load embeddings vào RAM ──
                cam_state["load_step"] = f"Đang tải dataset — {len(C._emb)} sinh viên..."
                cam_state["load_pct"]  = 15
                n_students = len(C._emb)
                log.info(f"[Warmup] Dataset: {n_students} sinh viên")
                time.sleep(0.3)   # nhường CPU cho response trả về trước

                # ── Bước 2: Warmup DeepFace model ──
                cam_state["load_step"] = "Đang khởi động model AI (lần đầu ~10s)..."
                cam_state["load_pct"]  = 30
                dummy = np.zeros((224, 224, 3), dtype=np.uint8)
                try:
                    C.recognize(dummy)   # gọi 1 lần để load model vào VRAM/RAM
                    log.info("[Warmup] Model AI loaded")
                except Exception as e:
                    log.warning(f"[Warmup] Dummy recognize: {e}")

                cam_state["load_pct"]  = 70
                cam_state["load_step"] = "Model sẵn sàng — Đang mở camera..."

                # ── Bước 3: Mở camera ──
                cap = cv2.VideoCapture(C.CAM_INDEX, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

                if not cap.isOpened():
                    log.error("[Warmup] Không mở được camera")
                    cam_state["loading"]   = False
                    cam_state["load_step"] = "❌ Lỗi: Không mở được camera!"
                    return

                cam_state["load_pct"]  = 90
                cam_state["load_step"] = "Camera đã mở — Đang lấy frame đầu tiên..."

                # Đọc vài frame để camera warm up (tránh frame tối đầu)
                for _ in range(5):
                    cap.read()

                # ── Bước 4: Bắt đầu stream ──
                cam_state["cap"]          = cap
                cam_state["session"]      = session_id
                cam_state["frame_counts"] = {}
                cam_state["confirmed"]    = {}
                cam_state["results"]      = []
                cam_state["frame"]        = None
                cam_state["running"]      = True
                cam_state["load_pct"]     = 100
                cam_state["load_step"]    = "✓ Sẵn sàng!"

                t = threading.Thread(target=_camera_loop, args=(C,), daemon=True)
                cam_state["thread"] = t
                t.start()

                log.info(f"[Warmup] Camera bật | buổi: {session_id} | {n_students} SV")

            finally:
                cam_state["loading"] = False

        threading.Thread(target=_start_async, daemon=True).start()
        return jsonify(ok=True, loading=True, session=session_id)

    @app.post("/api/stream/stop")
    def stream_stop():
        """Dừng camera."""
        cam_state["running"] = False
        return jsonify(ok=True)

    @app.get("/api/stream/status")
    def stream_status():
        """Trạng thái camera + loading progress + danh sách đã xác nhận."""
        confirmed = cam_state["confirmed"]
        fc_map    = cam_state["frame_counts"]
        now       = time.monotonic()
        return jsonify(
            ok        = True,
            running   = cam_state["running"],
            loading   = cam_state["loading"],
            load_step = cam_state["load_step"],
            load_pct  = cam_state["load_pct"],
            session   = cam_state["session"],
            confirmed = [
                { "sid": sid, **info }
                for sid, info in confirmed.items()
            ],
            pending   = [
                {
                    "sid":   sid,
                    "name":  v["name"],
                    "conf":  round(v["conf"] * 100, 1),
                    "elapsed": round(min(now - v["started_at"], 5.0), 1),
                    "remaining": round(max(0.0, 5.0 - (now - v["started_at"])), 1),
                    "pct":   min(round((now - v["started_at"]) / 5.0 * 100), 100),
                }
                for sid, v in fc_map.items()
                if sid not in confirmed
            ],
        )

    # ─────────────────────────────────────────────────────────
    #  THỐNG KÊ & DỮ LIỆU
    # ─────────────────────────────────────────────────────────

    @app.get("/api/stats")
    def get_stats():
        stats = C.db_stats()
        return jsonify(
            ok=True, data=stats,
            class_name=C.CLASS_NAME,
            session=datetime.now().strftime("%Y%m%d_%H%M"),
            time=datetime.now().strftime("%H:%M:%S"),
        )

    @app.get("/api/today")
    def get_today():
        rows = C.db_today()
        return jsonify(ok=True, data=rows, count=len(rows))

    @app.get("/api/attendance")
    def get_attendance():
        rows = C.db_attendance_history(
            filter_date    = request.args.get("date"),
            filter_session = request.args.get("session"),
        )
        return jsonify(ok=True, data=rows, count=len(rows))

    @app.get("/api/sessions")
    def get_sessions():
        return jsonify(ok=True, data=C.db_sessions())

    @app.get("/api/dataset")
    def get_dataset():
        students = C.db_students()
        result   = []
        for s in students:
            folder    = C.PHOTOS_DIR / s["id"]
            img_count = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
            trained   = s["id"] in C._emb
            result.append({**s, "img_count": img_count, "trained": trained})
        return jsonify(ok=True, data=result)

    # ─────────────────────────────────────────────────────────
    #  SINH VIÊN
    # ─────────────────────────────────────────────────────────

    @app.get("/api/students")
    def get_students():
        return jsonify(ok=True, data=C.db_students())

    @app.post("/api/students")
    def add_student():
        body = request.json or {}
        mssv = body.get("id",   "").strip()
        name = body.get("name", "").strip()
        cls  = body.get("cls",  "").strip()
        if not mssv or not name:
            return jsonify(ok=False, error="Thiếu MSSV hoặc Họ Tên"), 400
        C.db_add_student(mssv, name, cls)
        return jsonify(ok=True)

    @app.delete("/api/students/<mssv>")
    def delete_student(mssv: str):
        success = C.db_del_student(mssv)
        C._emb.pop(mssv, None)
        C._info.pop(mssv, None)
        return jsonify(ok=success)

    # ─────────────────────────────────────────────────────────
    #  TRAIN & EXPORT
    # ─────────────────────────────────────────────────────────

    @app.post("/api/train")
    def retrain():
        n = C.train_all()
        return jsonify(ok=True, count=n)

    @app.get("/api/export")
    def export_csv():
        session_id = request.args.get("session", datetime.now().strftime("%Y%m%d_%H%M"))
        rows       = C.db_session(session_id) or C.db_today()
        buf        = io.StringIO()
        buf.write("\ufeff")
        w = csv.DictWriter(buf, fieldnames=["id", "name", "time", "conf", "session"])
        w.writeheader()
        w.writerows(rows)
        b = io.BytesIO(buf.getvalue().encode("utf-8"))
        return send_file(b, mimetype="text/csv", as_attachment=True,
                         download_name=f"diemdanh_{session_id}.csv")

    return app
