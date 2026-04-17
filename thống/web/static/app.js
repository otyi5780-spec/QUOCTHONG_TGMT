// app.js — AttendAI Dashboard
// Nhóm 10 · Thị Giác Máy Tính · CV-CT4-K17

// ─────────────────────────────────────────────
// Tiện ích chung
// ─────────────────────────────────────────────

let histChart = null;
let methChart = null;
let allStudents = []; // Store students for client-side filtering

function toast(msg, type = "") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast show ${type}`;
  setTimeout(() => el.className = "toast", 3500);
}

async function api(url, opts = {}) {
  try {
    return await (await fetch(url, opts)).json();
  } catch {
    return { ok: false };
  }
}

// Đồng hồ
setInterval(() => {
  document.getElementById("clock").textContent =
    new Date().toLocaleTimeString("vi-VN");
}, 1000);


// ─────────────────────────────────────────────
// Tab routing
// ─────────────────────────────────────────────

const TAB_META = {
  dashboard :  ["Dashboard",         "Tổng quan hệ thống"],
  camera    :  ["Điểm Danh Camera",  "Nhận diện khuôn mặt qua webcam"],
  attendance:  ["Lịch Sử",           "Lịch sử điểm danh theo ngày / buổi"],
  dataset   :  ["Dataset",           "Sinh viên đã đăng ký ảnh khuôn mặt"],
  students  :  ["Sinh Viên",         "Quản lý danh sách sinh viên"],
};

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav").forEach(n => n.classList.remove("active"));

  document.getElementById(`tab-${name}`).classList.add("active");
  document.querySelector(`[data-tab="${name}"]`).classList.add("active");

  const [title, sub] = TAB_META[name] || ["", ""];
  document.getElementById("pageTitle").textContent = title;
  document.getElementById("pageSub").textContent   = sub;

  if (name === "attendance") { loadSessions(); loadAttendance(); }
  if (name === "students")   loadStudents();

  // Dừng camera nếu chuyển sang tab khác
  if (name !== "camera") stopCamera();
}

document.querySelectorAll(".nav[data-tab]").forEach(el => {
  el.addEventListener("click", () => switchTab(el.dataset.tab));
});


// ─────────────────────────────────────────────
// DASHBOARD
// ─────────────────────────────────────────────

async function loadStats() {
  const d = await api("/api/stats");
  const dotEl    = document.getElementById("dot");
  // Cập nhật khoảng CONFIRM
  const statusEl = document.getElementById("connStatus");

  if (!d.ok) {
    dotEl.className       = "dot off";
    statusEl.textContent  = "Mất kết nối";
    return;
  }
  dotEl.className       = "dot on";
  statusEl.textContent  = "Online";

  document.getElementById("cTotal").textContent = d.data.total_students ?? "—";
  document.getElementById("cToday").textContent = d.data.today_count    ?? "—";
  document.getElementById("cConf").textContent  = d.data.avg_conf ? `${d.data.avg_conf}%` : "—";
  document.getElementById("cFaces").textContent = d.data.registered_faces ?? "—";
  document.getElementById("sessionId").textContent  = d.session || "—";
  document.getElementById("className").textContent  = d.class_name || "";

  updateCharts(d.data);
}

function initCharts() {
  const ctxH = document.getElementById("historyChart")?.getContext("2d");
  const ctxM = document.getElementById("methodChart")?.getContext("2d");
  if (!ctxH || !ctxM) return;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      y: { beginAtZero: true, grid: { color: "#ffffff10" }, ticks: { color: "#8a97ab" } },
      x: { grid: { display: false }, ticks: { color: "#8a97ab" } }
    }
  };

  histChart = new Chart(ctxH, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Số lượng",
        data: [],
        borderColor: "#5b72f8",
        backgroundColor: "rgba(91, 114, 248, 0.1)",
        fill: true,
        tension: 0.4,
        borderWidth: 3,
        pointRadius: 4,
        pointBackgroundColor: "#5b72f8"
      }]
    },
    options: chartOptions
  });

  methChart = new Chart(ctxM, {
    type: "doughnut",
    data: {
      labels: ["Nhận diện mặt", "Thủ công"],
      datasets: [{
        data: [0, 0],
        backgroundColor: ["#22d3ee", "#5b72f8"],
        borderWidth: 0,
        hoverOffset: 10
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
         legend: { 
           position: "bottom",
           labels: { color: "#8a97ab", usePointStyle: true, padding: 20 }
         }
      },
      cutout: "70%"
    }
  });
}

function updateCharts(data) {
  if (!histChart || !methChart) initCharts();

  if (histChart && data.history) {
    histChart.data.labels = data.history.map(h => h.date);
    histChart.data.datasets[0].data = data.history.map(h => h.count);
    histChart.update();
  }

  if (methChart && data.method_stats) {
    methChart.data.datasets[0].data = [data.method_stats.face, data.method_stats.manual];
    methChart.update();
  }
}

async function loadToday() {
  const d = await api("/api/today");
  if (!d.ok) return;

  document.getElementById("todayCnt").textContent = d.count;
  const tbody = document.getElementById("bodyToday");

  tbody.innerHTML = d.data.length
    ? d.data.map(r => `
        <tr>
          <td><b>${r.id}</b></td>
          <td>${r.name}</td>
          <td><span style="color:var(--text3);font-size:.73rem">${r.session}</span></td>
          <td>${r.time}</td>
          <td>${r.conf ? r.conf + "%" : "—"}</td>
          <td>${methodChip(r.method)}</td>
        </tr>`).join("")
    : `<tr><td colspan="6" class="empty">Chưa có dữ liệu hôm nay</td></tr>`;
}

async function trainNow() {
  toast("⏳ Đang train...", "info");
  const d = await api("/api/train", { method: "POST" });
  d.ok ? toast(`✓ Train xong ${d.count} sinh viên`, "ok") : toast("✗ Train thất bại", "err");
  loadStats();
}

window.exportCSV = () => window.open("/api/export", "_blank");


// ─────────────────────────────────────────────
// CAMERA — MJPEG stream từ Python cv2
// ─────────────────────────────────────────────

let statusTimer = null;   // poll /api/stream/status

async function startStream() {
  const inp = document.getElementById("camSession");
  let session = inp?.value.trim();
  if (!session) {
    const now = new Date();
    const pad = n => String(n).padStart(2, "0");
    session = `${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
    if (inp) inp.value = session;
  }

  const d = await api("/api/stream/start", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ session }),
  });

  if (!d.ok) {
    toast("✗ " + (d.error || "Không bật được camera"), "err");
    return;
  }

  // Hiện loading overlay ngay lập tức
  showLoadingOverlay("Khởi động...", 0);
  document.getElementById("btnStartCam").style.display = "none";
  document.getElementById("btnStopCam").style.display  = "";

  // Poll trạng thái — thưa hơn trong lúc loading
  statusTimer = setInterval(pollStatus, 600);
  pollStatus();
}

async function stopStream() {
  await api("/api/stream/stop", { method: "POST" });
  clearInterval(statusTimer);
  statusTimer = null;

  // Tắt MJPEG + reset UI
  document.getElementById("camImg").src = "";
  hideLoadingOverlay();
  document.getElementById("camOffMsg").classList.remove("hidden");
  document.getElementById("camHud").style.display      = "none";
  document.getElementById("btnStartCam").style.display = "";
  document.getElementById("btnStopCam").style.display  = "none";

  toast("Camera đã dừng", "info");
  loadStats(); loadToday();
}

// ── Loading overlay helpers ──
function showLoadingOverlay(step, pct) {
  let overlay = document.getElementById("camLoadOverlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "camLoadOverlay";
    overlay.className = "cam-load-overlay";
    overlay.innerHTML = `
      <div class="cam-load-box">
        <div class="cam-load-spinner"></div>
        <div class="cam-load-title">⏳ Đang khởi động...</div>
        <div class="cam-load-step" id="camLoadStep"></div>
        <div class="cam-load-bar-wrap">
          <div class="cam-load-bar" id="camLoadBar"></div>
        </div>
        <div class="cam-load-pct" id="camLoadPct">0%</div>
      </div>`;
    document.getElementById("camViewport").appendChild(overlay);
  }
  overlay.style.display = "flex";
  document.getElementById("camLoadStep").textContent     = step;
  document.getElementById("camLoadBar").style.width      = pct + "%";
  document.getElementById("camLoadPct").textContent      = pct + "%";
}

function hideLoadingOverlay() {
  const el = document.getElementById("camLoadOverlay");
  if (el) el.style.display = "none";
}

async function pollStatus() {
  const d = await api("/api/stream/status");
  if (!d.ok) return;

  // ── Đang loading: hiện overlay tiến trình ──
  if (d.loading) {
    showLoadingOverlay(d.load_step || "Khởi động...", d.load_pct || 0);
    return;
  }

  // ── Load xong, camera đang chạy ──
  if (d.running && !d.loading) {
    hideLoadingOverlay();
    // Gán MJPEG nếu chưa gán
    const img = document.getElementById("camImg");
    if (!img.src || img.src.endsWith("/")) {
      img.src = "/api/stream";
    }
    document.getElementById("camOffMsg").classList.add("hidden");
    document.getElementById("camHud").style.display = "flex";
  }

  if (!d.running && statusTimer) {
    clearInterval(statusTimer);
    statusTimer = null;
  }

  // Cập nhật HUD
  const cnt = d.confirmed.length;
  document.getElementById("hudConfirmed").textContent = `✅ ${cnt} đã điểm danh`;
  const hudStatus = document.getElementById("hudStatus");
  if (d.pending.length > 0) {
    const best = d.pending.reduce((a, b) => a.pct > b.pct ? a : b);
    hudStatus.textContent = `⏳ ${best.name}: còn ${best.remaining.toFixed(1)}s`;
    hudStatus.className   = "hud-scanning";
  } else if (cnt > 0) {
    hudStatus.textContent = `✓ Tất cả đã xác nhận`;
    hudStatus.className   = "hud-found";
  } else {
    hudStatus.textContent = `🔍 Đang quét khuôn mặt...`;
    hudStatus.className   = "hud-scanning";
  }

  // Panel "Đã điểm danh"
  document.getElementById("camRecordCnt").textContent = cnt;
  const recList = document.getElementById("camResultList");
  recList.innerHTML = d.confirmed.length
    ? d.confirmed.map(s => `
        <div class="rec-card">
          <div class="rec-avatar">
            ${s.avatar ? `<img src="${s.avatar}" style="width:100%;height:100%;object-fit:cover;border-radius:50%">` : `✅`}
          </div>
          <div class="rec-info">
            <div class="rec-name">${s.name}</div>
            <div class="rec-meta">${s.sid} · Buổi: ${s.session}</div>
          </div>
          <div class="rec-time" style="text-align:right">
            <div>${s.time}</div>
            <div style="font-size:0.75rem;opacity:0.7">${s.date}</div>
          </div>
        </div>`).join("")
    : `<div class="cam-empty">Chưa xác nhận được ai</div>`;

  // Panel "Đang xác nhận"
  const pendList = document.getElementById("camFrameList");
  pendList.innerHTML = d.pending.length
    ? d.pending.map(p => `
        <div class="frame-chip known">
          <span>🟡</span>
          <span class="fc-name">${p.name}</span>
          <span class="fc-conf">${p.elapsed.toFixed(1)}/5.0s</span>
          <div class="fc-bar"><div class="fc-fill" style="width:${p.pct}%"></div></div>
          <div class="fc-meta">Giữ mặt thêm ${p.remaining.toFixed(1)}s · ${p.conf}%</div>
        </div>`).join("")
    : `<div class="cam-empty">Chưa phát hiện khuôn mặt cần giữ 5 giây</div>`;

  if (cnt > 0) { loadStats(); loadToday(); }
}

// Dừng stream khi chuyển tab
function stopCamera() { stopStream(); }
function startCamera() { startStream(); }


// ─────────────────────────────────────────────
// LỊCH SỬ ĐIỂM DANH
// ─────────────────────────────────────────────

async function loadSessions() {
  const d = await api("/api/sessions");
  if (!d.ok) return;

  const sel = document.getElementById("filterSession");
  const cur = sel.value;
  sel.innerHTML = `<option value="">-- Tất cả buổi --</option>` +
    d.data.map(s =>
      `<option value="${s.session}">${s.date} — ${s.session} (${s.count} SV)</option>`
    ).join("");
  if (cur) sel.value = cur;
}

async function loadAttendance() {
  const date    = document.getElementById("filterDate").value;
  const session = document.getElementById("filterSession").value;

  let url = "/api/attendance";
  const params = [];
  if (date)    params.push(`date=${date}`);
  if (session) params.push(`session=${encodeURIComponent(session)}`);
  if (params.length) url += "?" + params.join("&");

  const d = await api(url);
  if (!d.ok) return;

  document.getElementById("attCnt").textContent = d.count;
  const tbody = document.getElementById("bodyAtt");

  tbody.innerHTML = d.data.length
    ? d.data.map(r => `
        <tr>
          <td><b>${r.id}</b></td><td>${r.name}</td>
          <td>${r.date}</td><td>${r.time}</td>
          <td><span style="color:var(--text3);font-size:.73rem">${r.session}</span></td>
          <td>${r.conf ? r.conf + "%" : "—"}</td>
          <td>${methodChip(r.method)}</td>
        </tr>`).join("")
    : `<tr><td colspan="7" class="empty">Không có dữ liệu</td></tr>`;
}

function clearFilter() {
  document.getElementById("filterDate").value    = "";
  document.getElementById("filterSession").value = "";
  loadAttendance();
}


// ─────────────────────────────────────────────
// SINH VIÊN (COMBINED WITH DATASET)
// ─────────────────────────────────────────────

async function loadStudents() {
  const d = await api("/api/dataset"); // Sử dụng endpoint dataset để lấy đủ info
  if (!d.ok) return;

  allStudents = d.data;
  renderStudents(allStudents);
}

function renderStudents(list) {
  document.getElementById("svCnt").textContent = list.length;
  const tbody = document.getElementById("bodyStudents");

  tbody.innerHTML = list.length
    ? list.map(s => `
        <tr>
          <td><b>${s.id}</b></td>
          <td>${s.name}</td>
          <td>${s.cls}</td>
          <td>
            <span style="font-weight:700;color:${s.img_count >= 150 ? 'var(--green)' : 'var(--orange)'}">
              ${s.img_count}
            </span>
            <span style="color:var(--text3);font-size:.73rem"> / 150 ảnh</span>
          </td>
          <td>${s.trained
            ? `<span class="chip chip-ok">✓ Đã Train</span>`
            : `<span class="chip chip-miss">⚠ Chưa Train</span>`}
          </td>
          <td style="text-align:right">
            <button class="btn btn-sm btn-danger" onclick="delStudent('${s.id}','${s.name}')">
              Xoá
            </button>
          </td>
        </tr>`).join("")
    : `<tr><td colspan="6" class="empty">Không tìm thấy sinh viên nào</td></tr>`;
}

window.filterStudents = function() {
  const query = document.getElementById("svSearch").value.toLowerCase().trim();
  const filtered = allStudents.filter(s => 
    s.id.toLowerCase().includes(query) || 
    s.name.toLowerCase().includes(query)
  );
  renderStudents(filtered);
};

window.addStudent = async function(e) {
  e.preventDefault();
  const d = await api("/api/students", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      id:   document.getElementById("fid").value.trim(),
      name: document.getElementById("fn").value.trim(),
      cls:  document.getElementById("fc").value.trim(),
    }),
  });
  d.ok
    ? (toast("✓ Đã thêm sinh viên", "ok"), e.target.reset(), loadStudents())
    : toast("✗ " + (d.error || "Lỗi"), "err");
};

window.delStudent = async function(id, name) {
  if (!confirm(`Xoá sinh viên ${name} (${id})?`)) return;
  const d = await api(`/api/students/${id}`, { method: "DELETE" });
  d.ok ? (toast("Đã xoá", "ok"), loadStudents()) : toast("Lỗi xoá", "err");
};


// ─────────────────────────────────────────────
// Chip helper
// ─────────────────────────────────────────────

function methodChip(method) {
  return method === "face"
    ? `<span class="chip chip-face">🎯 Camera</span>`
    : `<span class="chip chip-manual">✋ Thủ công</span>`;
}


// ─────────────────────────────────────────────
// Auto-refresh
// ─────────────────────────────────────────────

async function refresh() {
  await Promise.all([loadStats(), loadToday()]);
}

refresh();
setInterval(refresh, 10000);
