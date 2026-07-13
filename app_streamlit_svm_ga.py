import streamlit as st
import numpy as np
import joblib
import os
from PIL import Image
import cv2
import pywt
from scipy.stats import skew, kurtosis
from skimage.restoration import denoise_tv_chambolle
import torchvision.transforms as transforms

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Klasifikasi Paru-paru SVM-GA", 
    page_icon="🫁", 
    layout="centered"
)

# --- CSS KHUSUS ---
st.markdown('''
    <style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-size: 1.1em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .result-box-abnormal {
        background-color: #ffeaea;
        border: 2px solid #ff4d4d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .result-text-abnormal {
        color: #cc0000;
        font-size: 1.8em;
        font-weight: bold;
        margin: 0;
    }
    .result-box-normal {
        background-color: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .result-text-normal {
        color: #27ae60;
        font-size: 1.8em;
        font-weight: bold;
        margin: 0;
    }
    .confidence-text {
        color: #555;
        font-size: 1em;
        margin-top: 10px;
    }
    </style>
''', unsafe_allow_html=True)

# --- HEADER APP ---
st.markdown("<div class='main-title'>🫁 Klasifikasi Kelainan Paru-Paru</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Support Vector Machine dengan Optimasi Genetika (SVM-GA)</div>", unsafe_allow_html=True)

# --- MEMUAT MODEL GA-SVM ---
@st.cache_resource
def load_models():
    model_dir = "saved_models/ga_optimized"
    svm_path = os.path.join(model_dir, "svm_model_ga_optimized.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    svm_model = joblib.load(svm_path) if os.path.exists(svm_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    return svm_model, scaler

svm_model, scaler = load_models()

# --- FUNGSI EKSTRAKSI & PREPROCESSING ---
def calculate_entropy(c):
    c_abs = np.abs(c)
    sum_c = np.sum(c_abs)
    if sum_c == 0:
        return 0
    else:
        p = c_abs / sum_c
        return -np.sum(p * np.log2(p + 1e-10))

def extract_features(image):
    '''
    Fungsi ini menggabungkan semua tahap preprocessing:
    1. Grayscale -> TV-Chambolle -> EqualizeHist
    2. Resize (240x240) -> ToTensor -> Normalize
    3. Ekstraksi DWT
    '''
    # --- TAHAP 1: PREPROCESSING AWAL (cv2 & skimage) ---
    # Konversi PIL Image ke array RGB lalu ke Grayscale via BGR (seperti cv2.imread)
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Anisotropic diffusion (TV-chambolle)
    diffused = denoise_tv_chambolle(gray, weight=0.1)
    diffused = (diffused * 255).astype(np.uint8)
    
    # Histogram equalization
    equalized = cv2.equalizeHist(diffused)
    
    # --- TAHAP 2: PREPROCESSING LANJUTAN (torchvision) ---
    # Konversi array kembali ke PIL untuk torchvision transforms
    equalized_pil = Image.fromarray(equalized)
    
    preprocessing_pipeline = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    preprocessed_tensor = preprocessing_pipeline(equalized_pil)
    # Ubah tensor kembali menjadi numpy array 2D (hilangkan dimensi channel/batch)
    final_img = preprocessed_tensor.numpy().squeeze()
    
    # --- TAHAP 3: EKSTRAKSI FITUR DWT ---
    feature_vector = []
    wavelet = 'haar'
    level = 1 

    # Perform DWT decomposition
    coeffs = pywt.wavedec2(final_img, wavelet, level=level)

    cA = coeffs[0]
    cH, cV, cD = coeffs[1]
    coefficients = {'LL': cA, 'LH': cH, 'HL': cV, 'HH': cD}

    for subband_name, subband in coefficients.items():
        c = subband.flatten()
        feature_vector.extend([
            np.mean(c),               # Mean
            np.var(c),                # Variance
            np.sum(c**2) / len(c),    # Energy
            calculate_entropy(c),     # Entropy
            skew(c),                  # Skewness
            kurtosis(c),              # Kurtosis
            np.max(c),                # Maximum
            np.min(c)                 # Minimum
        ])

    return np.array(feature_vector).reshape(1, -1)

# --- MAIN AREA ---
st.markdown("### Unggah Citra X-Ray Paru-paru")
uploaded_file = st.file_uploader("Seret dan lepas file di sini", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
    
    st.write("") 
    
    if st.button("🔍 Analisis Citra"):
        if svm_model is None:
            st.error("❌ Model SVM-GA tidak ditemukan. Pastikan folder 'saved_models/ga_optimized' berada pada direktori yang sama dengan aplikasi ini.")
        else:
            with st.spinner('Memproses gambar dan mengekstraksi fitur DWT...'):
                try:
                    # 1. Ekstraksi Fitur (sudah termasuk preprocessing lengkap)
                    raw_features = extract_features(image)
                    
                    # 2. Normalisasi Data (Scaler dari GA-SVM)
                    if scaler:
                        scaled_features = scaler.transform(raw_features)
                    else:
                        scaled_features = raw_features
                        
                    # 3. Prediksi Klasifikasi
                    prediction = svm_model.predict(scaled_features)
                    
                    # 4. Hitung Confidence (Tingkat Keyakinan)
                    confidence = 0.0
                    try:
                        # Jika SVM memiliki probability=True
                        probabilities = svm_model.predict_proba(scaled_features)[0]
                        confidence = max(probabilities) * 100
                    except AttributeError:
                        # Jika probability=False, gunakan decision_function
                        decision_scores = svm_model.decision_function(scaled_features)
                        if len(decision_scores.shape) > 1 or len(decision_scores) > 1: # Multi-class
                            decision_scores = decision_scores[0]
                            # Softmax heuristik
                            exp_scores = np.exp(decision_scores - np.max(decision_scores))
                            probs = exp_scores / np.sum(exp_scores)
                            confidence = np.max(probs) * 100
                        else: # Binary class
                            # Jarak mutlak ke hyperplane
                            dist = abs(decision_scores[0])
                            # Pemetaan heuristik sederhana (semakin jauh = semakin yakin)
                            confidence = min(100, max(50, 50 + (dist * 10)))

                    # 5. Cek Threshold untuk Memvalidasi Citra (Sesuaikan nilai jika perlu, e.g., 40.0 - 55.0)
                    CONFIDENCE_THRESHOLD = 50.0  
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning("⚠️ **Tingkat keyakinan model rendah** (" + f"{confidence:.2f}%" + ")")
                        st.error("Gambar yang Anda unggah kemungkinan besar **BUKAN citra Chest X-Ray** yang valid atau kualitas gambar terlalu buruk untuk dikenali.")
                        st.info("💡 Mohon ganti dan unggah citra Rontgen Dada (Chest X-Ray) yang sesuai.")
                    else:
                        # Pemetaan Kelas Prediksi 
                        class_labels = {
                            0: "COVID-19",
                            1: "Normal",
                            2: "Pneumonia Viral",
                            3: "Pneumonia Bakterial"
                        }
                        pred_label = prediction[0]
                        pred_class = class_labels.get(pred_label, f"Kelas {pred_label}")
                        
                        if pred_label == 1:
                            box_class = "result-box-normal"
                            text_class = "result-text-normal"
                            icon = "✅"
                        else:
                            box_class = "result-box-abnormal"
                            text_class = "result-text-abnormal"
                            icon = "⚠️"
                            
                        st.markdown(f'''
                            <div class="{box_class}">
                                <p class="confidence-text">Hasil Analisis:</p>
                                <p class="{text_class}">{icon} {pred_class}</p>
                                <p class="confidence-text">Tingkat Keyakinan (Akurasi Prediksi): <b>{confidence:.2f}%</b></p>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")