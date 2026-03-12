# Prediksi Penggunaan SPayLater Menggunakan Logistic Regression
Project ini bertujuan untuk menganalisis dan memprediksi kecenderungan penggunaan **SPayLater** dalam pembelian produk viral di media sosial.

Model yang digunakan adalah **Logistic Regression** dengan dua fitur utama:
- **FOMO Score**
- **Financial Literacy Score**

Dashboard dibuat menggunakan **Streamlit** untuk menampilkan:
- ringkasan data responden
- visualisasi perilaku penggunaan SPayLater
- simulasi prediksi
- insight model machine learning

## Tools
- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Cara Menjalankan
```bash
git clone https://github.com/dzakysyaam/PrediksiPenggunaanSPayLaterMenggunakanLogisticRegression.git
cd PrediksiPenggunaanSPayLaterMenggunakanLogisticRegression
pip install -r requirements.txt
streamlit run app.py
