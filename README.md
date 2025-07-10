# RiskGuard: Prediksi Biaya Asuransi dan Segmentasi Risiko Nasabah 🩺

Repositori ini berisi kode dan artefak untuk **RiskGuard**, sebuah proyek *machine learning* yang bertujuan memprediksi biaya klaim asuransi kesehatan dan melakukan segmentasi nasabah berdasarkan profil risiko mereka. Proyek ini diimplementasikan dalam sebuah *dashboard* interaktif menggunakan Streamlit.

## 📂 Struktur Repositori
.
├── models/                     # Folder berisi model dan artefak yang telah dilatih
│   ├── prediction_pipeline.joblib  # Pipeline lengkap untuk prediksi
│   ├── kmeans_model.joblib         # Model K-Means untuk segmentasi
│   └── clustering_scaler.joblib    # Scaler untuk data clustering
├── app.py                      # Kode aplikasi dashboard Streamlit
├── insurance_prediction.ipynb  # Notebook Jupyter berisi analisis dan pemodelan
└── README.md                   # Dokumentasi proyek (file ini)

## 🎯 Tujuan Proyek

Tujuan utama dari RiskGuard adalah untuk membantu tim *underwriting* dan pengembangan produk di industri asuransi dengan menyediakan alat bantu keputusan (*decision support tool*) yang cerdas. Platform ini mampu:
1.  **Memprediksi Biaya Tahunan**: Memberikan estimasi biaya klaim medis tahunan untuk nasabah baru berdasarkan profil mereka (umur, BMI, status merokok, dll.).
2.  **Segmentasi Risiko**: Secara otomatis mengklasifikasikan nasabah ke dalam segmen **'Low Risk'**, **'Medium Risk'**, atau **'High Risk'** menggunakan algoritma *unsupervised learning*.
3.  **Memberikan Rekomendasi**: Menyajikan opsi tindakan bisnis yang relevan untuk setiap segmen risiko, mengubah proses *underwriting* dari sekadar 'terima/tolak' menjadi manajemen risiko yang dipersonalisasi.

## 🛠️ Alur Kerja Teknis

Proyek ini dibangun melalui beberapa tahapan utama yang terdokumentasi dalam `insurance_prediction.ipynb`:

1.  **Analisis Data Eksploratif (EDA)**: Dataset awal (*Medical Cost Personal Dataset*) berisi 2.772 baris, namun setelah pembersihan data duplikat, tersisa 1.337 data unik. Analisis korelasi menunjukkan bahwa **status merokok**, **umur**, dan **BMI** adalah tiga faktor terkuat yang memengaruhi biaya asuransi.

2.  **Rekayasa Fitur (*Feature Engineering*)**:
    * **Fitur Kombinasi**: Membuat fitur seperti `smoker_and_obese` untuk menangkap interaksi risiko.
    * ***Unsupervised Clustering***: Menggunakan **K-Means** pada fitur `age`, `bmi`, dan `is_smoker` untuk menciptakan label segmen risiko.

3.  **Pemodelan (*Modelling*)**:
    * Tiga model regresi dibandingkan: **Linear Regression**, **Random Forest**, dan **Gradient Boosting**.
    * Berdasarkan metrik *Mean Absolute Error* (MAE) dan *R-squared* (R²), **Linear Regression** terpilih sebagai model terbaik dengan **MAE ~$2,419** dan **R² ~0.90**.

4.  **Penyimpanan Artefak**: Model prediksi, model K-Means, dan *scaler* disimpan sebagai file `.joblib` di dalam folder `models/` untuk digunakan oleh aplikasi Streamlit.

## 🚀 Cara Menjalankan Aplikasi

Aplikasi *dashboard* interaktif ini dibangun menggunakan **Streamlit**. Pastikan Anda telah menginstal semua *library* yang dibutuhkan sebelum menjalankan.

### Prasyarat

Anda membutuhkan Python 3.8+ dan *library* berikut. Anda dapat menginstalnya menggunakan pip:
```bash
pip install streamlit pandas scikit-learn joblib
streamlit run app.py