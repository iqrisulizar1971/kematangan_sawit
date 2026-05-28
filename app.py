"""
╔══════════════════════════════════════════════════════════════╗
║   SISTEM DETEKSI KEMATANGAN BUAH SAWIT — YOLOv8              ║
║   Streamlit Cloud Ready · Auto-download model via GDrive     ║
║   Social Investment Indonesia (SII)                          ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# SUPPRESS WARNINGS — harus sebelum import lain
# ─────────────────────────────────────────────
import warnings
import logging
import os

# Matikan warning "missing ScriptRunContext" dari Ultralytics/YOLO
# yang muncul saat library logging diinisialisasi sebelum Streamlit context siap
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Matikan verbose logging Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("ultralytics.utils.torch_utils").setLevel(logging.ERROR)

# Sembunyikan output YOLO ke terminal
os.environ["YOLO_VERBOSE"] = "False"

# ─────────────────────────────────────────────
# IMPORT UTAMA
# ─────────────────────────────────────────────
import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
import time, io, tempfile

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN (harus paling atas)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Kematangan Sawit | SII",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# KONSTANTA — SESUAIKAN DI SINI
# ─────────────────────────────────────────────
# Ganti FILE_ID dengan ID file best.pt di Google Drive Anda
# Cara ambil: klik kanan file → "Bagikan" → salin link
# Contoh link: https://drive.google.com/file/d/1AbCdEfGhIj.../view
# https://drive.google.com/file/d/1Ua3yuEsX3bDhiVSYLiS-Ka7A78vxdC-M/view
# FILE_ID = "1Ua3yuEsX3bDhiVSYLiS-Ka7A78vxdC-M"   <── ganti ini
GDRIVE_FILE_ID  = "1Ua3yuEsX3bDhiVSYLiS-Ka7A78vxdC-M"
MODEL_LOCAL_PATH = "weights/best.pt"

# Nama kelas — sesuaikan dengan data.yaml Anda
LABEL_MAP = {
    0: {"nama": "Mentah",       "en": "Unripe",    "warna": (33,  101, 192), "emoji": "🟢"},
    1: {"nama": "Matang",       "en": "Ripe",       "warna": (198, 40,  40), "emoji": "🔴"},
    2: {"nama": "Lewat Matang", "en": "Overripe",   "warna": (106, 27, 154), "emoji": "🟣"},
}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,#1a5c2a 0%,#2d8a47 55%,#f9a825 100%);
    padding: 1.4rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.15);
}
.main-header h1 { color:#fff!important; font-size:1.8rem!important;
    margin:0!important; font-weight:700; text-shadow:1px 1px 4px rgba(0,0,0,.3); }
.main-header p  { color:rgba(255,255,255,.9)!important; margin:.3rem 0 0!important; font-size:.9rem; }
.metric-card { background:#fff; border-radius:12px; padding:1rem 1.2rem;
    text-align:center; box-shadow:0 2px 12px rgba(0,0,0,.08);
    border-left:5px solid #4caf50; margin-bottom:.6rem; }
.metric-card.kuning { border-left-color:#f9a825; }
.metric-card.merah  { border-left-color:#c62828; }
.metric-card h3 { font-size:1.9rem; margin:0; color:#333; font-weight:700; }
.metric-card p  { margin:0; font-size:.82rem; color:#666; font-weight:500; }
.info-box { background:#e8f5e9; border:1px solid #a5d6a7; border-radius:10px;
    padding:.9rem 1.1rem; margin:.4rem 0; font-size:.88rem; color:#2e7d32; }
.warn-box { background:#fff8e1; border:1px solid #ffcc02; border-radius:10px;
    padding:.9rem 1.1rem; margin:.4rem 0; font-size:.88rem; color:#f57f17; }
.step-box { background:#f3f6ff; border-left:4px solid #3f51b5; border-radius:8px;
    padding:.8rem 1rem; margin:.5rem 0; font-size:.88rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a5c2a 0%,#2d8a47 100%); }
section[data-testid="stSidebar"] * { color:#fff!important; }
section[data-testid="stSidebar"] input { color:#111!important; background:#fff!important; border-radius:6px; }
section[data-testid="stSidebar"] textarea { color:#111!important; background:#fff!important; }
.stTabs [data-baseweb="tab"] {
    border-radius:8px 8px 0 0; padding:8px 18px;
    background:#e8f5e9; color:#2e7d32!important; font-weight:600; }
.stTabs [aria-selected="true"] { background:#2d8a47!important; color:#fff!important; }
.stButton>button { background:linear-gradient(135deg,#2d8a47,#1a5c2a);
    color:#fff; border:none; border-radius:10px; padding:.55rem 1.8rem;
    font-weight:600; font-size:.95rem; width:100%; transition:all .2s; }
.stButton>button:hover { transform:translateY(-1px);
    box-shadow:0 4px 15px rgba(45,138,71,.4); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FUNGSI: DOWNLOAD MODEL DARI GOOGLE DRIVE
# ─────────────────────────────────────────────
def download_model_gdrive(file_id: str, dest: str) -> bool:
    """Download model dari Google Drive pakai gdown."""
    try:
        import gdown
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("⏳ Mengunduh model dari Google Drive... (sekali saja)"):
            gdown.download(url, dest, quiet=False)
        return Path(dest).exists() and Path(dest).stat().st_size > 1_000_000
    except Exception as e:
        st.error(f"❌ Gagal mengunduh model: {e}")
        return False

# ─────────────────────────────────────────────
# FUNGSI: LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Memuat model YOLOv8...")
def load_model(path: str) -> YOLO:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = YOLO(path)
    return model

# ─────────────────────────────────────────────
# FUNGSI: CEK PERANGKAT
# ─────────────────────────────────────────────
def cek_perangkat() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", f"🖥️ GPU: {torch.cuda.get_device_name(0)}"
    return "cpu", "💻 CPU (Streamlit Cloud — normal)"

# ─────────────────────────────────────────────
# FUNGSI: INFERENSI
# ─────────────────────────────────────────────
def inferensi(model, img_bgr, conf, iou, device):
    results   = model.predict(source=img_bgr, conf=conf, iou=iou,
                               device=device, verbose=False)
    result    = results[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    deteksi   = []
    if result.boxes is not None:
        for box in result.boxes:
            cid  = int(box.cls.item())
            cf   = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            info = LABEL_MAP.get(cid, {"nama": f"Kelas {cid}", "emoji": "⚪"})
            deteksi.append({
                "No"         : len(deteksi) + 1,
                "Kelas"      : info["nama"],
                "Emoji"      : info["emoji"],
                "Kepercayaan": round(cf * 100, 1),
                "X1": int(xyxy[0]), "Y1": int(xyxy[1]),
                "X2": int(xyxy[2]), "Y2": int(xyxy[3]),
            })
    return annotated, deteksi

# ─────────────────────────────────────────────
# FUNGSI: GRAFIK
# ─────────────────────────────────────────────
def grafik_bar(deteksi):
    df  = pd.DataFrame(deteksi)
    cnt = df["Kelas"].value_counts().reset_index()
    cnt.columns = ["Kelas", "Jumlah"]
    warna = {"Mentah":"#1565c0","Matang":"#c62828","Lewat Matang":"#6a1b9a"}
    fig = px.bar(cnt, x="Kelas", y="Jumlah", color="Kelas",
                 color_discrete_map=warna, text="Jumlah",
                 title="Distribusi Kelas Deteksi")
    fig.update_layout(showlegend=False, height=260, margin=dict(t=40,b=10,l=0,r=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_traces(textposition="outside")
    return fig

def grafik_gauge(conf_rata):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf_rata,
        title={"text": "Rata-rata Kepercayaan (%)"},
        gauge={"axis":{"range":[0,100]}, "bar":{"color":"#2d8a47"},
               "steps":[{"range":[0,50],"color":"#ffcdd2"},
                        {"range":[50,75],"color":"#fff9c4"},
                        {"range":[75,100],"color":"#c8e6c9"}],
               "threshold":{"line":{"color":"#1a5c2a","width":3},"value":80}},
        number={"suffix":"%","font":{"size":26}},
    ))
    fig.update_layout(height=210, margin=dict(t=30,b=5,l=15,r=15),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def tampilkan_tabel(deteksi):
    if not deteksi:
        st.info("Tidak ada objek terdeteksi.")
        return
    df = pd.DataFrame(deteksi)[["No","Emoji","Kelas","Kepercayaan","X1","Y1","X2","Y2"]]
    df["Kepercayaan"] = df["Kepercayaan"].astype(str) + " %"
    df = df.rename(columns={"Emoji":"","Kepercayaan":"Conf (%)"})
    st.dataframe(df.set_index("No"), use_container_width=True)

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌴 Konfigurasi")
    st.markdown("---")

    # ── Sumber model ──
    st.markdown("### 📂 Model")
    sumber_model = st.radio(
        "Sumber model:",
        ["Google Drive (otomatis)", "Upload manual (.pt)"],
        index=0
    )

    model = None

    if sumber_model == "Google Drive (otomatis)":
        gdrive_id = st.text_input(
            "Google Drive File ID",
            value=GDRIVE_FILE_ID,
            help="ID dari link share Google Drive file best.pt"
        )
        if st.button("⬇️ Download & Muat Model"):
            if gdrive_id == "GANTI_DENGAN_FILE_ID_ANDA" or len(gdrive_id) < 10:
                st.error("Masukkan File ID Google Drive yang valid.")
            elif Path(MODEL_LOCAL_PATH).exists():
                st.success("✅ Model sudah ada, langsung dimuat.")
                model = load_model(MODEL_LOCAL_PATH)
            else:
                ok = download_model_gdrive(gdrive_id, MODEL_LOCAL_PATH)
                if ok:
                    st.success("✅ Model berhasil diunduh!")
                    model = load_model(MODEL_LOCAL_PATH)
        # Coba muat otomatis jika sudah ada
        if model is None and Path(MODEL_LOCAL_PATH).exists():
            model = load_model(MODEL_LOCAL_PATH)
            st.success("✅ Model dimuat dari cache lokal.")

    else:
        uploaded_pt = st.file_uploader("Upload file .pt", type=["pt"])
        if uploaded_pt:
            Path("weights").mkdir(exist_ok=True)
            with open(MODEL_LOCAL_PATH, "wb") as f:
                f.write(uploaded_pt.read())
            model = load_model(MODEL_LOCAL_PATH)
            st.success("✅ Model berhasil dimuat!")

    st.markdown("---")

    # ── Parameter ──
    st.markdown("### ⚙️ Parameter Deteksi")
    conf_thr = st.slider("Confidence Threshold", 0.10, 0.95, 0.50, 0.05)
    iou_thr  = st.slider("IoU Threshold (NMS)",  0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    device_val, device_lbl = cek_perangkat()
    st.markdown(f"### 💻 Perangkat\n**{device_lbl}**")

    st.markdown("---")
    st.markdown("### 🎨 Legenda Kelas")
    for cid, info in LABEL_MAP.items():
        r, g, b = info["warna"]
        st.markdown(
            f"{info['emoji']} "
            f"<span style='background:rgb({r},{g},{b});color:white;"
            f"padding:2px 9px;border-radius:12px;font-size:.8rem;font-weight:600'>"
            f"{info['nama']}</span>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("🌴 SII · YOLOv8 Palm Ripeness")

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🌴 Sistem Deteksi Kematangan Buah Sawit</h1>
  <p>Iqri Sulizar Hidriansjah · Kontak WA: 0852-1939-0680 . Email: iqrisulizar@gmail.com . LinkedIn: linkedin.com/in/iqrisulizar</p>
</div>
""", unsafe_allow_html=True)

# Status model
if model is None:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Model belum dimuat.</strong> Silakan pilih sumber model di sidebar kiri, 
    lalu klik <strong>"Download & Muat Model"</strong> atau upload file <code>.pt</code> langsung.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="info-box">✅ Model siap digunakan. Upload gambar, video, atau gunakan kamera untuk memulai deteksi.</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# TAB NAVIGASI
# ═══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️  Gambar",
    "🎥  Video",
    "📷  Kamera",
    "ℹ️  Panduan Deploy",
])

# ══════════════════════════════════════════════
# TAB 1 — GAMBAR
# ══════════════════════════════════════════════
with tab1:
    st.markdown("#### Upload gambar buah sawit (bisa lebih dari satu)")

    uploaded_imgs = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1: show_bbox  = st.checkbox("Tampilkan Bounding Box", True)
    with col_o2: show_table = st.checkbox("Tampilkan Tabel", True)
    with col_o3: export_csv = st.checkbox("Ekspor CSV Rekap", False)

    if uploaded_imgs:
        if model is None:
            st.error("❌ Muat model terlebih dahulu di sidebar.")
        else:
            semua_det = []
            bar = st.progress(0, text="Memproses...")

            for idx, f in enumerate(uploaded_imgs):
                raw   = np.frombuffer(f.read(), np.uint8)
                bgr   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                t0 = time.perf_counter()
                annotated, det = inferensi(model, bgr, conf_thr, iou_thr, device_val)
                ms = (time.perf_counter() - t0) * 1000

                for d in det:
                    d["File"] = f.name
                semua_det.extend(det)

                with st.container():
                    st.markdown(f"---\n##### 📄 `{f.name}`")
                    c_in, c_out = st.columns(2, gap="medium")
                    with c_in:
                        st.markdown("**📥 Input**")
                        st.image(rgb, use_container_width=True)
                        st.caption(f"Ukuran: {rgb.shape[1]} × {rgb.shape[0]} px")
                    with c_out:
                        st.markdown("**📤 Output Deteksi**")
                        st.image(annotated if show_bbox else rgb, use_container_width=True)
                        st.caption(f"⏱️ {ms:.0f} ms · {len(det)} objek terdeteksi")

                    # Metrik
                    m1, m2, m3, m4 = st.columns(4)
                    kls_cnt = pd.DataFrame(det)["Kelas"].value_counts().to_dict() if det else {}
                    conf_r  = np.mean([d["Kepercayaan"] for d in det]) if det else 0
                    with m1:
                        st.markdown(f'<div class="metric-card"><h3>{len(det)}</h3><p>Total</p></div>',
                                    unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card merah"><h3>{kls_cnt.get("Matang",0)}</h3><p>🔴 Matang</p></div>',
                                    unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card"><h3>{kls_cnt.get("Mentah",0)}</h3><p>🟢 Mentah</p></div>',
                                    unsafe_allow_html=True)
                    with m4:
                        st.markdown(f'<div class="metric-card kuning"><h3>{conf_r:.0f}%</h3><p>Avg Conf</p></div>',
                                    unsafe_allow_html=True)

                    if det:
                        g1, g2 = st.columns([3,2])
                        with g1: st.plotly_chart(grafik_bar(det), use_container_width=True)
                        with g2: st.plotly_chart(grafik_gauge(conf_r), use_container_width=True)

                    if show_table:
                        tampilkan_tabel(det)

                    # Download gambar hasil
                    buf = io.BytesIO()
                    Image.fromarray(annotated).save(buf, format="JPEG", quality=92)
                    st.download_button(
                        f"⬇️ Unduh Hasil — {f.name}",
                        data=buf.getvalue(),
                        file_name=f"hasil_{f.name}",
                        mime="image/jpeg",
                        key=f"dl_{idx}",
                    )

                bar.progress((idx+1)/len(uploaded_imgs),
                             text=f"{idx+1}/{len(uploaded_imgs)} gambar diproses")

            bar.empty()

            # Rekap batch
            if len(uploaded_imgs) > 1 and semua_det:
                st.markdown("---\n### 📊 Rekap Semua Gambar")
                df_all = pd.DataFrame(semua_det)
                st.dataframe(
                    df_all[["File","Emoji","Kelas","Kepercayaan"]].rename(
                        columns={"Emoji":"","Kepercayaan":"Conf (%)"}
                    ),
                    use_container_width=True
                )
                if export_csv:
                    csv_bytes = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Unduh CSV Rekap",
                        data=csv_bytes,
                        file_name=f"rekap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

# ══════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════
with tab2:
    st.markdown("#### Upload video sawit untuk dideteksi frame per frame")
    st.markdown('<div class="info-box">💡 Di Streamlit Cloud, proses video berjalan di CPU — cocok untuk video pendek (&lt; 1 menit).</div>', unsafe_allow_html=True)

    vid_file = st.file_uploader("Pilih video", type=["mp4","avi","mov","mkv"],
                                 label_visibility="collapsed")

    cv1, cv2c, cv3 = st.columns(3)
    with cv1: n_frame  = st.number_input("Proses setiap N frame", 1, 30, 5)
    with cv2c: maks_frm = st.number_input("Maks. frame diproses",  10, 300, 60)
    with cv3: show_prev = st.checkbox("Tampilkan preview", True)

    if vid_file and st.button("▶️ Mulai Deteksi Video", use_container_width=True):
        if model is None:
            st.error("❌ Muat model terlebih dahulu.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            cap    = cv2.VideoCapture(tmp_path)
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_v  = cap.get(cv2.CAP_PROP_FPS) or 30
            st.info(f"📹 {total} frame · {fps_v:.1f} FPS")

            prev_ph = st.empty()
            stat_ph = st.empty()
            bar_v   = st.progress(0, text="Memproses video...")

            fno, pno, all_det, ms_list = 0, 0, [], []

            while cap.isOpened() and pno < maks_frm:
                ret, frame = cap.read()
                if not ret: break
                if fno % n_frame == 0:
                    t0 = time.perf_counter()
                    ann, det = inferensi(model, frame, conf_thr, iou_thr, device_val)
                    ms_list.append((time.perf_counter()-t0)*1000)
                    for d in det: d["Frame"] = fno
                    all_det.extend(det)
                    if show_prev:
                        prev_ph.image(ann, caption=f"Frame #{fno}", use_container_width=True)
                    stat_ph.metric("Frame diproses", pno+1)
                    pno += 1
                fno += 1
                bar_v.progress(min(pno/maks_frm, 1.0), text=f"Frame {fno}/{total}")

            cap.release()
            os.unlink(tmp_path)
            bar_v.empty()

            st.success(f"✅ {pno} frame selesai · {len(all_det)} total deteksi")

            if all_det:
                df_v  = pd.DataFrame(all_det)
                tren  = df_v.groupby(["Frame","Kelas"]).size().reset_index(name="Jumlah")
                fig_t = px.line(tren, x="Frame", y="Jumlah", color="Kelas",
                                title="Tren Deteksi per Frame",
                                color_discrete_map={"Matang":"#c62828","Mentah":"#1565c0","Lewat Matang":"#6a1b9a"})
                fig_t.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_t, use_container_width=True)

                csv_v = df_v.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Unduh CSV Hasil Video", csv_v,
                    file_name=f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv")

# ══════════════════════════════════════════════
# TAB 3 — KAMERA
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### Foto langsung dari kamera HP / webcam")
    st.markdown("""
    <div class="info-box">
    📱 <strong>Cara pakai di HP Android:</strong><br>
    Buka aplikasi ini di browser Chrome → tap <em>"Upload Gambar"</em> di Tab Gambar → 
    pilih <strong>"Ambil Foto"</strong> (ikon kamera) → foto tandan sawit → deteksi otomatis berjalan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Atau gunakan kamera komputer (jika akses lokal):**")

    col_c1, col_c2 = st.columns([1,2])
    with col_c1:
        idx_cam = st.number_input("Indeks kamera", 0, 5, 0)
        ambil   = st.button("📸 Ambil Snapshot Kamera", use_container_width=True)

    with col_c2:
        cam_ph = st.empty()
        if ambil:
            if model is None:
                st.error("❌ Muat model terlebih dahulu.")
            else:
                cam = cv2.VideoCapture(idx_cam)
                if not cam.isOpened():
                    cam_ph.error(
                        f"Kamera {idx_cam} tidak dapat diakses.\n\n"
                        "**Di Streamlit Cloud** kamera lokal tidak tersedia — "
                        "gunakan Tab Gambar dan upload foto dari galeri HP."
                    )
                else:
                    for _ in range(5): cam.read()
                    ret, frm = cam.read()
                    cam.release()
                    if ret:
                        ann, det = inferensi(model, frm, conf_thr, iou_thr, device_val)
                        ca, cb = cam_ph.columns(2)
                        with ca: st.image(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB),
                                           caption="Gambar Kamera", use_container_width=True)
                        with cb: st.image(ann, caption="Hasil Deteksi", use_container_width=True)
                        tampilkan_tabel(det)
                    else:
                        cam_ph.error("Gagal membaca frame kamera.")
        else:
            cam_ph.markdown("""
            <div style="height:220px;display:flex;align-items:center;justify-content:center;
            background:#f0f7f0;border-radius:12px;border:2px dashed #2d8a47;flex-direction:column">
                <span style="font-size:2.5rem">📷</span>
                <p style="color:#2d8a47;font-weight:600;margin:.4rem 0 0">
                Klik tombol untuk ambil snapshot</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — PANDUAN DEPLOY
# ══════════════════════════════════════════════
with tab4:
    st.markdown("#### Panduan Deploy ke Streamlit Cloud")

    st.markdown("""
    <div class="step-box">
    <strong>Struktur folder yang harus diupload ke GitHub:</strong>
    </div>

    ```
    repo-sawit-sii/
    ├── app.py                ← file ini
    ├── requirements.txt
    ├── .streamlit/
    │   └── config.toml
    └── weights/              ← kosongkan, model didownload otomatis
        └── .gitkeep
    ```
    """, unsafe_allow_html=True)

    st.markdown("---")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("""
        **① Upload Model ke Google Drive**
        1. Buka [drive.google.com](https://drive.google.com)
        2. Upload file `best.pt`
        3. Klik kanan → **Bagikan**
        4. Ubah akses ke **"Siapa saja yang punya link"**
        5. Salin link, ambil **File ID** dari URL:
        ```
        .../file/d/**1AbCdEfGhIj**/view
                     ^^^^^^^^^^^^^^
                     ini File ID-nya
        ```
        6. Tempel di sidebar app
        """)

    with step2:
        st.markdown("""
        **② Push ke GitHub**
        ```bash
        git init
        git add .
        git commit -m "init sawit app"
        git remote add origin \
          https://github.com/username/repo
        git push -u origin main
        ```
        Pastikan `weights/best.pt` ada di `.gitignore`
        agar tidak ikut terupload (terlalu besar).
        """)

    with step3:
        st.markdown("""
        **③ Deploy di Streamlit Cloud**
        1. Buka [share.streamlit.io](https://share.streamlit.io)
        2. Login dengan GitHub
        3. Klik **"New app"**
        4. Pilih repo & branch
        5. Main file path: `app.py`
        6. Klik **"Deploy!"**
        7. Tunggu ~3–5 menit
        8. Dapat URL publik ✅

        URL bisa langsung dibuka dari HP Android!
        """)

    st.markdown("---")
    st.markdown("**`requirements.txt` yang dibutuhkan:**")
    st.code("""streamlit>=1.35.0
ultralytics>=8.2.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
gdown>=5.1.0""", language="text")

    st.markdown("**`.streamlit/config.toml`:**")
    st.code("""[server]
maxUploadSize = 200

[theme]
primaryColor = "#2d8a47"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f7f0"
textColor = "#1a1a1a"
font = "sans serif"
""", language="toml")

    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Catatan performa di Streamlit Cloud (gratis):</strong><br>
    • RAM: ~1 GB — cukup untuk YOLOv8n/s/m<br>
    • CPU only — inferensi ~1–3 detik/gambar (wajar)<br>
    • Sleep otomatis jika tidak aktif 7 hari (wake up saat dibuka kembali)<br>
    • Untuk produksi lebih cepat → upgrade ke VPS + GPU
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:.8rem;color:#888;font-size:.82rem">
    🌴 <strong>Sistem Deteksi Kematangan Buah Sawit</strong> · Iqri Sulizar Hidriansjah<br>
    Bisa diakses dari HP Android via browser Chrome 📱
</div>
""", unsafe_allow_html=True)
