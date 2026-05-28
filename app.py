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
    """Download model dari Google Drive/OneDrive pakai gdown."""
    try:
        import gdown
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("⏳ Mengunduh model dari Google Drive/OneDrive ... (sekali saja)"):
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
        ["Google Drive/OneDrive (otomatis)", "Upload manual (.pt)"],
        index=0
    )

    model = None

    if sumber_model == "Google Drive/OneDrive (otomatis)":
        gdrive_id = st.text_input(
            "",
            value=GDRIVE_FILE_ID,
            help="ID dari link share Google Drive file best.pt"
        )
        if st.button("⬇️ Download & Muat Model"):
            if gdrive_id == "GANTI_DENGAN_FILE_ID_ANDA" or len(gdrive_id) < 10:
                st.error("Masukkan File ID Google Drive/OneDrive yang valid.")
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
  <p>Iqri Sulizar Hidriansjah · Kontak WA: 0852-1939-0680 . Email: iqrisulizar@socialinvestment.id . LinkedIn: linkedin.com/in/iqrisulizar</p>
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
    "👤  Tentang Kami",
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
# TAB 4 — TENTANG KAMI
# ══════════════════════════════════════════════
with tab4:

    FOTO_B64 = "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"

    # ── Hero Banner ──


    col_profil, col_detail = st.columns([1, 2], gap="large")

    with col_profil:
        st.html(f"""
<div style="background:#fff;border-radius:16px;padding:1.8rem;
box-shadow:0 2px 16px rgba(0,0,0,.08);text-align:center">
    <img src="data:image/jpeg;base64,{FOTO_B64}"
         style="width:160px;height:160px;object-fit:cover;object-position:top center;
         border-radius:50%;border:4px solid #2d8a47;
         box-shadow:0 4px 16px rgba(45,138,71,.25);margin-bottom:.8rem;
         background:#f0f7f0"/>
    <h3 style="margin:0;color:#1a5c2a;font-size:1.1rem;font-weight:700">
        Iqri Sulizar Hidriansjah</h3>
    <p style="margin:.3rem 0 .8rem;color:#666;font-size:.83rem">
        Konsultan Strategi dan Keberlanjutan<br>Praktisi Teknologi Informasi</p>
    <div style="background:#e8f5e9;border-radius:8px;padding:.45rem;
    font-size:.8rem;color:#2e7d32;margin-bottom:.8rem">
        &#128205; Bogor, Jawa Barat, Indonesia
    </div>
    <a href="https://linkedin.com/in/iqrisulizar" target="_blank"
    style="background:#0077b5;color:white;padding:.4rem 1.2rem;
    border-radius:8px;text-decoration:none;font-size:.82rem;font-weight:600;
    display:inline-block">
        &#128279; LinkedIn
    </a>
</div>
""")

        st.html("<br>")

        st.html("""
<div style="background:#fff;border-radius:16px;padding:1.4rem;
box-shadow:0 2px 16px rgba(0,0,0,.08)">
    <h4 style="color:#1a5c2a;margin:0 0 .8rem;font-size:.95rem">&#127891; Pendidikan</h4>
    <div style="border-left:3px solid #2d8a47;padding-left:.8rem;margin-bottom:.8rem">
        <p style="margin:0;font-weight:700;font-size:.88rem;color:#333">
            Magister Teknik Informatika (S2)</p>
        <p style="margin:.1rem 0;font-size:.82rem;color:#2d8a47;font-weight:600">
            Universitas Pamulang</p>
        <p style="margin:0;font-size:.8rem;color:#888">Jan 2024 &#8211; 2026 (Perkiraan)</p>
    </div>
    <div style="background:#f3f6ff;border-radius:8px;padding:.7rem;font-size:.8rem;color:#3f51b5">
        &#128221; <strong>Tesis:</strong> Kerangka Analitik Multimodel untuk Otomatisasi
        Analisis Materialitas Ganda pada Laporan Keberlanjutan menggunakan
        BERT, Bi-LSTM dan TextGCN
    </div>
</div>
""")

        st.html("<br>")

        st.html("""
<div style="background:#fff;border-radius:16px;padding:1.4rem;
box-shadow:0 2px 16px rgba(0,0,0,.08)">
    <h4 style="color:#1a5c2a;margin:0 0 .8rem;font-size:.95rem">&#128270; Minat Riset</h4>
    <ul style="margin:0;padding-left:1.1rem;font-size:.82rem;color:#555;line-height:1.8">
        <li>Pengembangan Aplikasi untuk Investasi Sosial</li>
        <li>Otomasi alur kerja dan arsitektur dokumen digital untuk efisiensi korporat</li>
    </ul>
</div>
""")

    with col_detail:
        # Ringkasan profil
        st.html("""
<div style="background:#fff;border-radius:16px;padding:1.5rem;
box-shadow:0 2px 16px rgba(0,0,0,.08);margin-bottom:1rem">
    <h4 style="color:#1a5c2a;margin:0 0 .8rem">&#128203; Ringkasan Profil</h4>
    <p style="color:#444;font-size:.9rem;line-height:1.75;margin:0">
    Profesional yang menggabungkan keahlian mendalam di bidang
    <strong>keberlanjutan perusahaan (Sustainability)</strong> dan
    <strong>teknologi informasi modern</strong>. Berpengalaman sebagai konsultan
    strategi PROPER dan pengukur dampak sosial (SROI), serta aktif mengintegrasikan
    kerangka analitik berbasis <strong>Kecerdasan Buatan</strong>
    (NLP dan Deep Learning) dalam pelaporan keberlanjutan (ESG).
    Memiliki rekam jejak yang kuat dalam arsitektur digital, otomatisasi proses
    (Microsoft 365), dan pengelolaan ekosistem akademik.
    </p>
</div>
""")

        # Pengalaman
        st.html("""
<div style="background:#fff;border-radius:16px;padding:1.5rem;
box-shadow:0 2px 16px rgba(0,0,0,.08);margin-bottom:1rem">
    <h4 style="color:#1a5c2a;margin:0 0 1rem">&#128188; Pengalaman Profesional</h4>

    <div style="border-left:4px solid #2d8a47;padding-left:1rem;margin-bottom:1rem">
        <p style="margin:0;font-weight:700;color:#333;font-size:.9rem">
            Konsultan Investasi Sosial</p>
        <p style="margin:.1rem 0 .2rem;font-size:.82rem;color:#2d8a47;font-weight:600">
            Social Investment Indonesia (SII)</p>
        <p style="margin:0 0 .5rem;font-size:.8rem;color:#888">2021 &#8211; Sekarang</p>
        <ul style="margin:0;padding-left:1.2rem;font-size:.85rem;color:#555;line-height:1.75">
            <li>Mengembangkan dan mengimplementasikan aplikasi untuk Investasi Sosial (SLO, SLA, IKM, SROI, dll)</li>
            <li>Mengelola pengembangan SROI App (sroi.socialinvestment.id)</li>
            <li>Mengimplementasikan model Kecerdasan Buatan (AI) untuk Layanan Publik dan Bisnis</li>
        </ul>
    </div>

    <div style="border-left:4px solid #f9a825;padding-left:1rem;margin-bottom:1rem">
        <p style="margin:0;font-weight:700;color:#333;font-size:.9rem">
            Manajer Inisiatif TI dan Operasional</p>
        <p style="margin:.1rem 0 .2rem;font-size:.82rem;color:#f9a825;font-weight:600">
            Arasis Awan Teknologi</p>
        <p style="margin:0 0 .5rem;font-size:.8rem;color:#888">2022 &#8211; Sekarang</p>
        <ul style="margin:0;padding-left:1.2rem;font-size:.85rem;color:#555;line-height:1.75">
            <li>Memimpin inisiatif teknologi dan mengelola platform digital arasis.id</li>
        </ul>
    </div>

    <div style="border-left:4px solid #1565c0;padding-left:1rem">
        <p style="margin:0;font-weight:700;color:#333;font-size:.9rem">
            Administrator Sistem Jurnal</p>
        <p style="margin:.1rem 0 .2rem;font-size:.82rem;color:#1565c0;font-weight:600">
            Jurnal Canting</p>
        <p style="margin:0 0 .5rem;font-size:.8rem;color:#888">2023 &#8211; Sekarang</p>
        <ul style="margin:0;padding-left:1.2rem;font-size:.85rem;color:#555;line-height:1.75">
            <li>Mengelola platform jurnal ilmiah (canting.socialinvestment.id)</li>
            <li>Menyelesaikan permasalahan teknis dalam pengelolaan jurnal canting</li>
        </ul>
    </div>
</div>
""")

        # Keahlian
        st.html("""
<div style="background:#fff;border-radius:16px;padding:1.5rem;
box-shadow:0 2px 16px rgba(0,0,0,.08)">
    <h4 style="color:#1a5c2a;margin:0 0 1rem">&#9889; Keahlian dan Teknologi</h4>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
        <div style="flex:1;min-width:180px;background:#e8f5e9;border-radius:10px;padding:1rem">
            <p style="margin:0 0 .6rem;font-weight:700;color:#1a5c2a;font-size:.88rem">
                &#127807; Keberlanjutan dan Strategi</p>
            <ul style="margin:0;padding-left:1.1rem;font-size:.83rem;color:#444;line-height:1.8">
                <li>Social Return on Investment (SROI)</li>
                <li>PROPER (Kementerian LHK)</li>
                <li>Pelaporan ESG dan GRI 400 Series</li>
                <li>Kepatuhan POJK 51/2017</li>
                <li>Analisis Materialitas Ganda</li>
            </ul>
        </div>
        <div style="flex:1;min-width:180px;background:#e3f2fd;border-radius:10px;padding:1rem">
            <p style="margin:0 0 .6rem;font-weight:700;color:#1565c0;font-size:.88rem">
                &#129302; Teknologi dan AI</p>
            <ul style="margin:0;padding-left:1.1rem;font-size:.83rem;color:#444;line-height:1.8">
                <li>NLP dan Deep Learning: BERT, Bi-LSTM, TextGCN</li>
                <li>Computer Vision: Klasifikasi dan Deteksi</li>
                <li>Microsoft 365: SharePoint, Power Automate, Power Apps</li>
                <li>Riset Data: NVivo, OJS</li>
                <li>Flutter &#183; Streamlit &#183; Python</li>
            </ul>
        </div>
    </div>
</div>
""")

    # ── Tentang Aplikasi ──
    st.markdown("---")
    st.markdown("""
    <div style="background:#fff;border-radius:16px;padding:1.5rem;
    box-shadow:0 2px 16px rgba(0,0,0,.08)">
        <h4 style="color:#1a5c2a;margin:0 0 1rem">&#127796; Tentang Aplikasi Ini</h4>
        <div style="display:flex;gap:1rem;flex-wrap:wrap">
            <div style="flex:1;min-width:200px;background:#f9fbe7;border-radius:10px;
            padding:1rem;border-left:4px solid #827717">
                <p style="margin:0 0 .3rem;font-weight:700;font-size:.88rem;color:#827717">&#127919; Tujuan</p>
                <p style="margin:0;font-size:.83rem;color:#555;line-height:1.6">
                Mengotomatisasi penilaian kematangan tandan buah segar (TBS) sawit
                menggunakan model YOLOv8 untuk mendukung efisiensi panen di sektor perkebunan.</p>
            </div>
            <div style="flex:1;min-width:200px;background:#fce4ec;border-radius:10px;
            padding:1rem;border-left:4px solid #c62828">
                <p style="margin:0 0 .3rem;font-weight:700;font-size:.88rem;color:#c62828">&#128300; Model</p>
                <p style="margin:0;font-size:.83rem;color:#555;line-height:1.6">
                YOLOv8 dilatih dengan dataset gabungan, punya kemmpuan mendeteksi 3 kelas: Mentah, Matang, dan Lewat Matang.</p>
            </div>
            <div style="flex:1;min-width:200px;background:#e8eaf6;border-radius:10px;
            padding:1rem;border-left:4px solid #3f51b5">
                <p style="margin:0 0 .3rem;font-weight:700;font-size:.88rem;color:#3f51b5">&#128241; Akses</p>
                <p style="margin:0;font-size:.83rem;color:#555;line-height:1.6">
                Dapat diakses dari HP Android via browser Chrome tanpa instalasi apapun.
                Deploy di Streamlit Cloud dengan model tersimpan di Google Drive/OneDrive.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:.8rem;color:#888;font-size:.82rem">
    🌴 <strong>Sistem Deteksi Kematangan Buah Sawit</strong> · Iqri Sulizar Hidriansjah<br>
    Bisa diakses dari HP Android via browser Chrome 📱
</div>
""", unsafe_allow_html=True)
