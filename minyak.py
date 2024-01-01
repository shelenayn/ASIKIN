import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model_path = "modelasik4.h5"  # Ganti dengant path model Anda

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_my_model(model_path):
    model = load_model(model_path)
    return model

# Fungsi untuk melakukan prediksi
def import_and_predict(image_data, loaded_model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)

    # Konversi gambar ke format RGB
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = loaded_model.predict(img_reshape)

    return prediction

# Tampilan web menggunakan Streamlit
st.title("Klasifikasi Tumpahan Minyak")
st.write("Klasifikasi ini menggunakan algoritma CNN untuk mengklasifikasi ada atau tidaknya tumpahan minyak")

# Upload gambar dari user
file = st.file_uploader("Pilih file", type=["jpg", "png"])

# Jika ada file yang diupload
if file is not None:
    # Membaca gambar
    image = Image.open(file)
    st.image(image, caption="Gambar yang Diunggah.", use_column_width=True)

    # Load model
    model = load_my_model(model_path)

    # Prediksi kelas gambar
    predictions = import_and_predict(image, model)
    class_names = ["Tedeteksi ada tumpahan minyak", "Terdeteksi tidak ada tumpahan minyak"] 

    # Menampilkan hasil prediksi
    st.write("Prediksi:")
    st.write(class_names[np.argmax(predictions)])