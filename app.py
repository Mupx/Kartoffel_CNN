import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Lade das Modell (gecached um Ladezeiten zu sparen)
@st.cache_resource
def load_keras_model():
    model = load_model('model_1.keras')
    return model

model = load_keras_model()

st.title("Bildklassifizierung mit Keras und Streamlit")
uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "png", "jpeg"])

st.title("Foto aufnehmen statt hochladen")

# Kamera-Input
picture = st.camera_input(label="Camera", help="Kamera erlauben und auf 'Take Photo' klicken")

if picture:
    st.success("Foto erfolgreich aufgenommen!")

    img = Image.open(picture)

    st.image(img, caption="Dein aufgenommenes Foto")

    if img is not None:
        st.write("")
        st.write("Klassifiziere...")

    # Bild für das Modell vorbereiten
    image = img.resize((224, 224)) # Bildgroeße für das Modell
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

        
    # Vorhersage
    prediction = model.predict(image_array)

    class_names = ['Kartoffel: Early Blight', 'Kartoffel: Late Blight', 'Kartoffel: Gesund'] 

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    st.success(f"Vorhersage: {predicted_class}")

 
    confidence = np.max(prediction)
    st.write(f"Konfidenz: {confidence*100:.2f}%")

else:
    st.info("Warte auf die Aufnahme eines Fotos...")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
    st.write("")
    st.write("Klassifiziere...")

    # Bild für das Modell vorbereiten
    image = image.resize((224, 224)) # Bildgroeße für das Modell
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Vorhersage
    prediction = model.predict(image_array)

    class_names = ['Kartoffel: Early Blight', 'Kartoffel: Late Blight', 'Kartoffel: Gesund'] 

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    st.success(f"Vorhersage: {predicted_class}")

    confidence = np.max(prediction)
    st.write(f"Konfidenz: {confidence*100:.2f}%")