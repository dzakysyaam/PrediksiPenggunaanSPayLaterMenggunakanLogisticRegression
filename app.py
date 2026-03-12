import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIG
st.set_page_config(
    page_title="SPayLater Analytics Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PATH
MODEL_PATH = "artifacts/spaylater_logreg.pkl"
DATA_PATH = "artifacts/dashboard_spaylater.csv"
METRICS_PATH = "artifacts/metrics.json"

# VALIDATION FILE
required_files = [MODEL_PATH, DATA_PATH, METRICS_PATH]

for path in required_files:
    if not os.path.exists(path):
        st.error(f"File belum ditemukan: {path}")
        st.info("Pastikan file model, dataset dashboard, dan metrics sudah tersedia.")
        st.stop()

# LOADERS
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
df = load_data()
metrics = load_metrics()

# REQUIRED COLUMNS CHECK
required_columns = [
    "fomo_score",
    "financial_score",
    "target_viral_food",
    "fashion",
    "viral_food",
    "skincare",
    "hiburan",
    "tagihan",
    "gender",
    "status"
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Kolom pada dataset belum lengkap: {missing_cols}")
    st.stop()

# HELPER
def score_label(score):
    if score < 2.5:
        return "Rendah"
    elif score < 3.5:
        return "Sedang"
    return "Tinggi"

def probability_label(prob):
    if prob < 0.40:
        return "Rendah"
    elif prob < 0.70:
        return "Sedang"
    return "Tinggi"

def metric_delta_label(value, average):
    diff = round(value - average, 2)
    if diff > 0:
        return f"+{diff} di atas rata-rata"
    elif diff < 0:
        return f"{diff} di bawah rata-rata"
    return "Sama dengan rata-rata"

avg_fomo = round(df["fomo_score"].mean(), 2)
avg_financial = round(df["financial_score"].mean(), 2)
avg_target = round(df["target_viral_food"].mean() * 100, 1)
total_responden = len(df)

# CUSTOM CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border-right: 1px solid #e9eef5;
    }

    .main-title-box {
        background: linear-gradient(135deg, #5B5FEF 0%, #2F80ED 100%);
        border-radius: 24px;
        padding: 30px 32px;
        color: white;
        box-shadow: 0 12px 35px rgba(47, 128, 237, 0.20);
        margin-bottom: 18px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .main-subtitle {
        font-size: 15px;
        opacity: 0.96;
        line-height: 1.7;
    }

    .info-strip {
        background: #ffffff;
        border: 1px solid #e8edf5;
        border-radius: 18px;
        padding: 14px 18px;
        box-shadow: 0 4px 18px rgba(17, 24, 39, 0.05);
        margin-bottom: 18px;
    }

    .section-heading {
        font-size: 22px;
        font-weight: 800;
        color: #111827;
        margin-top: 6px;
        margin-bottom: 12px;
    }

    .premium-card {
        background: #ffffff;
        border: 1px solid #e8edf5;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.05);
        margin-bottom: 14px;
    }

    .premium-card-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 10px;
        font-weight: 600;
    }

    .premium-card-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }

    .premium-card-sub {
        font-size: 13px;
        color: #6b7280;
        margin-top: 6px;
    }

    .soft-chip {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
        border: 1px solid transparent;
    }

    .chip-fomo {
        background: #efe9ff;
        color: #5b3fd6;
        border-color: #ddd6fe;
    }

    .chip-finance {
        background: #e8f7ef;
        color: #1f8b4c;
        border-color: #bbf7d0;
    }

    .chip-model {
        background: #e8f0ff;
        color: #2d63d6;
        border-color: #bfdbfe;
    }

    .story-card {
        background: #ffffff;
        border: 1px solid #e9edf5;
        border-left: 6px solid #5B5FEF;
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 5px 18px rgba(17, 24, 39, 0.04);
        margin-bottom: 12px;
    }

    .result-success {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #d1fae5;
        border-radius: 18px;
        padding: 18px;
        font-weight: 700;
        margin-top: 8px;
    }

    .result-warning {
        background: #fff7ed;
        color: #c2410c;
        border: 1px solid #ffedd5;
        border-radius: 18px;
        padding: 18px;
        font-weight: 700;
        margin-top: 8px;
    }

    .result-neutral {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #dbeafe;
        border-radius: 18px;
        padding: 18px;
        margin-top: 8px;
    }

    .mini-label {
        font-size: 12px;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: 6px;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg, #5B5FEF 0%, #2F80ED 100%);
        color: white;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(47, 128, 237, 0.18);
    }

    .stButton > button:hover {
        opacity: 0.96;
    }

    .footer-note {
        color: #6b7280;
        font-size: 13px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("## 💳 SPayLater Analytics")
st.sidebar.caption("Dashboard prediksi pembelian produk viral")

page = st.sidebar.radio(
    "Navigasi",
    ["Overview", "Prediction Studio", "Model Insight", "Project Notes"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Fokus Analisis")
st.sidebar.markdown(
    """
    <div style="margin-bottom:10px;">
        <div class="mini-label">Faktor Analisis</div>
        <span class="soft-chip chip-fomo">FOMO</span>
        <span class="soft-chip chip-finance">Literasi Keuangan</span>
    </div>
    <div style="margin-top:6px;">
        <div class="mini-label">Metode</div>
        <span class="soft-chip chip-model">Logistic Regression</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard ini membantu menjelaskan hubungan antara perilaku FOMO, kontrol finansial, "
    "dan kecenderungan penggunaan SPayLater untuk pembelian produk viral."
)

# HERO
st.markdown("""
<div class="main-title-box">
    <div class="main-title">SPayLater Consumer Behavior Dashboard</div>
    <div class="main-subtitle">
        Dashboard ini dirancang untuk menganalisis hubungan antara <b>FOMO</b>,
        <b>literasi keuangan</b>, dan kemungkinan penggunaan <b>SPayLater</b>
        pada pembelian produk viral di media sosial menggunakan model
        <b>Logistic Regression</b>.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="info-strip">
    <b>Snapshot Analisis:</b> {total_responden} responden |
    Rata-rata FOMO <b>{avg_fomo}</b> |
    Rata-rata Literasi Keuangan <b>{avg_financial}</b> |
    Target penggunaan produk viral <b>{avg_target}%</b>
</div>
""", unsafe_allow_html=True)

# PAGE: OVERVIEW
if page == "Overview":
    st.markdown('<div class="section-heading">Executive Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-card-title">Jumlah Responden</div>
            <div class="premium-card-value">{total_responden}</div>
            <div class="premium-card-sub">Dataset valid untuk analisis dashboard</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-card-title">Rata-rata FOMO</div>
            <div class="premium-card-value">{avg_fomo}</div>
            <div class="premium-card-sub">Kategori: {score_label(avg_fomo)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-card-title">Literasi Keuangan</div>
            <div class="premium-card-value">{avg_financial}</div>
            <div class="premium-card-sub">Kategori: {score_label(avg_financial)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-card-title">Viral Purchase Rate</div>
            <div class="premium-card-value">{avg_target}%</div>
            <div class="premium-card-sub">Proporsi target = Ya</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Behavior Analytics</div>', unsafe_allow_html=True)

    purchase_counts = pd.DataFrame({
        "Kategori": ["Fashion", "Produk Viral", "Skincare", "Hiburan", "Tagihan"],
        "Jumlah Ya": [
            int(df["fashion"].sum()),
            int(df["viral_food"].sum()),
            int(df["skincare"].sum()),
            int(df["hiburan"].sum()),
            int(df["tagihan"].sum())
        ]
    })

    fig_bar = px.bar(
        purchase_counts,
        x="Kategori",
        y="Jumlah Ya",
        text="Jumlah Ya",
        title="Kategori Pembelian yang Pernah Menggunakan SPayLater"
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=430,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        gender_counts = df["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Jumlah"]

        fig_gender = px.pie(
            gender_counts,
            names="Gender",
            values="Jumlah",
            hole=0.55,
            title="Distribusi Gender Responden"
        )
        fig_gender.update_layout(
            height=380,
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    with col_right:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Jumlah"]

        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Jumlah",
            text="Jumlah",
            title="Distribusi Status Responden"
        )
        fig_status.update_traces(textposition="outside")
        fig_status.update_layout(
            height=380,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    scatter_df = df.copy()
    scatter_df["Target"] = scatter_df["target_viral_food"].map({1: "Ya", 0: "Tidak"})

    fig_scatter = px.scatter(
        scatter_df,
        x="fomo_score",
        y="financial_score",
        color="Target",
        title="Sebaran FOMO vs Literasi Keuangan",
        hover_data=["gender", "status"]
    )
    fig_scatter.update_layout(
        height=440,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-heading">Alur Penjelasan Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="story-card">
        <b>1. Mulai dari responden</b><br>
        Dashboard dibuka dengan ringkasan jumlah responden dan karakteristik dasarnya
        agar pengguna memahami konteks data terlebih dahulu.
    </div>

    <div class="story-card">
        <b>2. Masuk ke perilaku penggunaan</b><br>
        Setelah itu, dashboard menunjukkan kategori pembelian yang paling sering
        menggunakan SPayLater sebagai dasar analisis perilaku konsumsi.
    </div>

    <div class="story-card">
        <b>3. Hubungkan dengan faktor utama</b><br>
        Fokus analisis diarahkan pada dua variabel utama, yaitu FOMO dan literasi keuangan,
        sebagai fitur yang memengaruhi kecenderungan penggunaan SPayLater.
    </div>

    <div class="story-card">
        <b>4. Tutup dengan simulasi prediksi</b><br>
        Pada tahap akhir, pengguna dapat melakukan simulasi skor dan melihat hasil prediksi
        menggunakan model Logistic Regression.
    </div>
    """, unsafe_allow_html=True)

# PAGE: PREDICTION STUDIO
elif page == "Prediction Studio":
    st.markdown('<div class="section-heading">Prediction Studio</div>', unsafe_allow_html=True)
    st.caption("Isi penilaian 1 sampai 5 untuk melihat kecenderungan penggunaan SPayLater.")

    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown("#### Faktor FOMO")
        q1 = st.slider("Saya merasa khawatir jika tertinggal tren yang sedang populer di media sosial.", 1, 5, 3)
        q2 = st.slider("Saya tertarik membeli produk yang sedang viral di media sosial.", 1, 5, 3)
        q3 = st.slider("Saya cenderung ingin membeli suatu barang setelah melihat orang lain memilikinya.", 1, 5, 3)
        q4 = st.slider("Saya merasa takut kehilangan kesempatan ketika ada promo atau diskon online.", 1, 5, 3)

        st.markdown("#### Faktor Literasi Keuangan")
        q5 = st.slider("Saya menyisihkan sebagian pendapatan saya untuk ditabung.", 1, 5, 3)
        q6 = st.slider("Saya membuat perencanaan keuangan sebelum melakukan pembelian.", 1, 5, 3)
        q7 = st.slider("Saya mempertimbangkan kondisi keuangan sebelum menggunakan layanan paylater.", 1, 5, 3)
        q8 = st.slider("Saya berusaha menabung secara rutin setiap bulan.", 1, 5, 3)
        q9 = st.slider("Saya menghindari pembelian barang yang tidak terlalu dibutuhkan.", 1, 5, 3)

    fomo_score = round((q1 + q2 + q3 + q4) / 4, 2)
    financial_score = round((q5 + q6 + q7 + q8 + q9) / 5, 2)

    with right_col:
        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-card-title">Skor FOMO Kamu</div>
            <div class="premium-card-value">{fomo_score}</div>
            <div class="premium-card-sub">
                Kategori: {score_label(fomo_score)} | {metric_delta_label(fomo_score, avg_fomo)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(fomo_score / 5, 0.0), 1.0))

        st.markdown(f"""
        <div class="premium-card">
            <div class="premium-card-title">Skor Literasi Keuangan</div>
            <div class="premium-card-value">{financial_score}</div>
            <div class="premium-card-sub">
                Kategori: {score_label(financial_score)} | {metric_delta_label(financial_score, avg_financial)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(min(max(financial_score / 5, 0.0), 1.0))

        st.markdown("""
        <div class="premium-card">
            <div class="premium-card-title">Catatan Analisis</div>
            <div class="premium-card-sub" style="font-size:14px; line-height:1.7;">
                Semakin tinggi skor FOMO, semakin besar kemungkinan pengguna terdorong
                membeli produk viral. Sebaliknya, literasi keuangan yang lebih baik
                cenderung menekan keputusan pembelian impulsif.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Generate Prediction Result"):
        input_df = pd.DataFrame({
            "fomo_score": [fomo_score],
            "financial_score": [financial_score]
        })

        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])

        st.markdown("### Prediction Output")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Probabilitas Menggunakan SPayLater"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 40], "color": "#dbeafe"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#fecaca"}
                ]
            }
        ))
        gauge.update_layout(height=320)
        st.plotly_chart(gauge, use_container_width=True)

        result_cols = st.columns(2)

        with result_cols[0]:
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="result-warning">
                        Hasil Prediksi: Cenderung menggunakan SPayLater untuk membeli produk viral.<br>
                        Probabilitas: {prob:.2%} ({probability_label(prob)})
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-success">
                        Hasil Prediksi: Cenderung tidak menggunakan SPayLater untuk membeli produk viral.<br>
                        Probabilitas: {prob:.2%} ({probability_label(prob)})
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with result_cols[1]:
            delta_fomo = round(fomo_score - avg_fomo, 2)
            delta_fin = round(financial_score - avg_financial, 2)

            st.markdown(
                f"""
                <div class="result-neutral">
                    <b>Interpretasi Singkat</b><br><br>
                    • Skor FOMO kamu <b>{'lebih tinggi' if delta_fomo > 0 else 'lebih rendah' if delta_fomo < 0 else 'setara'}</b> dibanding rata-rata responden.<br>
                    • Literasi keuangan kamu <b>{'lebih tinggi' if delta_fin > 0 else 'lebih rendah' if delta_fin < 0 else 'setara'}</b> dibanding rata-rata responden.<br>
                    • Model menggabungkan dua skor ini untuk membaca kecenderungan penggunaan SPayLater.
                </div>
                """,
                unsafe_allow_html=True
            )

# PAGE: MODEL INSIGHT
elif page == "Model Insight":
    st.markdown('<div class="section-heading">Model Insight</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with m2:
        st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Aktual Tidak", "Aktual Ya"],
        columns=["Prediksi Tidak", "Prediksi Ya"]
    )

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

    with right:
        coef_df = pd.DataFrame({
            "Fitur": ["FOMO Score", "Financial Score"],
            "Koefisien": [
                metrics["coefficients"]["fomo_score"],
                metrics["coefficients"]["financial_score"]
            ]
        })

        fig_coef = px.bar(
            coef_df,
            x="Fitur",
            y="Koefisien",
            text="Koefisien",
            title="Pengaruh Masing-masing Fitur"
        )
        fig_coef.update_traces(textposition="outside")
        fig_coef.update_layout(
            height=360,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    fomo_coef = metrics["coefficients"]["fomo_score"]
    fin_coef = metrics["coefficients"]["financial_score"]

    st.markdown("### Penjelasan Model")
    st.markdown(f"""
    <div class="story-card">
        <b>Koefisien FOMO = {fomo_coef:.4f}</b><br>
        Nilai positif menunjukkan bahwa semakin tinggi FOMO,
        maka kecenderungan menggunakan SPayLater juga meningkat.
    </div>
    <div class="story-card">
        <b>Koefisien Literasi Keuangan = {fin_coef:.4f}</b><br>
        Nilai negatif menunjukkan bahwa semakin baik literasi keuangan,
        maka kecenderungan penggunaan SPayLater cenderung menurun.
    </div>
    <div class="story-card">
        <b>Kesimpulan Insight</b><br>
        Dalam dataset ini, FOMO berperan sebagai pendorong utama perilaku penggunaan SPayLater,
        sedangkan literasi keuangan bertindak sebagai faktor pengendali.
    </div>
    """, unsafe_allow_html=True)

# PAGE: PROJECT NOTES
elif page == "Project Notes":
    st.markdown('<div class="section-heading">Project Notes</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="story-card">
        <b>Tujuan Project</b><br>
        Memprediksi kecenderungan penggunaan SPayLater untuk pembelian produk viral
        dengan mempertimbangkan aspek psikologis dan finansial.
    </div>
    <div class="story-card">
        <b>Metode yang Digunakan</b><br>
        Logistic Regression dipilih karena target bersifat biner:
        menggunakan atau tidak menggunakan SPayLater.
    </div>
    <div class="story-card">
        <b>Fitur Utama</b><br>
        1. FOMO Score<br>
        2. Financial Score
    </div>
    <div class="story-card">
        <b>Target Prediksi</b><br>
        Penggunaan SPayLater untuk pembelian makanan atau minuman viral di media sosial.
    </div>
    <div class="story-card">
        <b>Nilai Tambah Dashboard</b><br>
        Dashboard ini tidak hanya menampilkan hasil prediksi, tetapi juga membantu
        menjelaskan alur analisis secara visual dan mudah dipahami.
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown('<div class="footer-note">Syam Developer • SPayLater Behavioral Analytics</div>', unsafe_allow_html=True)