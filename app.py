import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load
@st.cache_resource
def load_models():
    prediction_pipeline = joblib.load(os.path.join('models', 'prediction_pipeline.joblib'))
    kmeans_model = joblib.load(os.path.join('models', 'kmeans_model.joblib'))
    clustering_scaler = joblib.load(os.path.join('models', 'clustering_scaler.joblib'))
    return prediction_pipeline, kmeans_model, clustering_scaler

prediction_pipeline, kmeans, scaler = load_models()


# Prediction
def predict_risk(data):
    df = data.copy()
    
    def get_bmi_cat(bmi) :
        if bmi < 18.5 : return "Underweight"
        elif 18.5 <= bmi < 25 : return "Normal"
        elif 25 <= bmi < 30 : return "Overweight"
        else : return "Obese"

    df['bmi_category'] = df['bmi'].apply(get_bmi_cat)
    df['is_smoker'] = (df['smoker'] == 'no').astype(int)
    df['is_obese'] = (df['bmi_category'] == 'Obese').astype(int)
    df['smoker_and_obese'] = df['is_smoker'] * df['is_obese']
    

    # Feature For Clustering
    features_for_clustering = df[['age', 'bmi', 'is_smoker']]
    scaled_features = scaler.transform(features_for_clustering)
    risk_segment = kmeans.predict(scaled_features)[0]
    segment_map = {0: 'Low Risk', 1: 'High Risk', 2: 'Medium Risk'}
    risk_label = segment_map.get(risk_segment, 'Unknown')
    
    # Pipeline Prediction
    df['risk_segment_label'] = risk_label
    prediction = prediction_pipeline.predict(df)[0]
    
    return prediction, risk_label

# UI
st.set_page_config(page_title="RiskGuard Dashboard", layout="wide")
st.title("RiskGuard: Platform Analisis Risiko Nasabah Asuransi 🩺")

st.sidebar.header("Masukkan Data Nasabah Baru")

age = st.sidebar.number_input("Umur", 18, 100, 30)
sex = st.sidebar.selectbox("Jenis Kelamin", ['male', 'female'])
bmi = st.sidebar.number_input("Body Mass Index (BMI)", 15.0, 60.0, 25.0)
children = st.sidebar.number_input("Jumlah Anak", 0, 10, 0)
smoker = st.sidebar.selectbox("Apakah Perokok?", ['no', 'yes'])
region = st.sidebar.selectbox("Wilayah", ['southwest', 'southeast', 'northwest', 'northeast'])

if st.sidebar.button("Analisis Risiko"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    predicted_charge, risk_segment = predict_risk(input_data)
    
    st.subheader("Hasil Analisis Risiko")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediksi Biaya Tahunan", f"${predicted_charge:,.2f}")
    with col2:
        st.metric("Segmen Risiko", risk_segment)
        
    st.markdown("---")
    
    st.subheader("Rekomendasi Tindakan Bisnis")
    if risk_segment == 'High Risk':
        st.warning("⚠️ **Risiko Tinggi Terdeteksi**")
        st.write("Nasabah ini memiliki profil risiko yang sangat tinggi. Pertimbangkan untuk menawarkan:\n" \
        "- **Paket 'Premi Cerdas'**: Sesuaikan premi secara akurat untuk mencerminkan risiko yang ada.\n" \
        "- **Paket 'Mitra Sehat'**: Tawarkan paket premium yang mencakup benefit proaktif seperti konsultasi gizi atau keanggotaan gym\n" \
        "- **Wajib Pemeriksaan Awal**: Lakukan pemeriksaan kesehatan sebelum polis diaktifkan untuk validasi risiko.")
    elif risk_segment == 'Medium Risk':
        st.info("💡 **Risiko Sedang Terdeteksi**")
        st.write("Nasabah ini memiliki beberapa faktor risiko. Pertimbangkan untuk menawarkan:\n" \
        "- **Paket 'Jaga Sehat'**: Tawarkan paket standar dengan *add-on* opsional untuk program pencegahan.\n" \
        "- **Program Insentif**: Berikan diskon premi tahunan jika nasabah berhasil mencapai target kesehatan (misal: menurunkan BMI).")
    else: # Low Risk
        st.success("✅ **Risiko Rendah Terdeteksi**")
        st.write("Profil nasabah ini sangat baik. Pertimbangkan untuk menawarkan:\n" \
        "- **Paket 'Premi Terbaik'**: Berikan harga paling kompetitif untuk menarik dan mempertahankan segmen nasabah sehat.\n" \
        "- **Proses Cepat**: Tawarkan proses pendaftaran yang paling mudah dan cepat.")