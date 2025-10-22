import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Modell Laden ---
# Modell aendern falls nötig
MODEL = 'normal_CNN.keras'

# Lade das Modell (gecached um Ladezeiten zu sparen)
@st.cache_resource
def load_keras_model():
    """Lädt das kompilierte Keras-Modell."""
    try:
        model = load_model(MODEL)
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        st.error("Stelle sicher, dass die Datei {MODEL} im selben Verzeichnis wie die App liegt.")
        return None

model = load_keras_model()

# --- App-Konfiguration ---
st.set_page_config(page_title="Pflanzen-Klassifizierung", layout="wide")

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Gesund']

# Bilder müssen auf GitHub liegen!
RESULT_IMAGE_MAP = {
    'Early Blight': 'https://github.com/schmidleonard/Kartoffel_CNN/blob/main/Bilder/bauer2.png?raw=true',
    'Late Blight': 'https://github.com/schmidleonard/Kartoffel_CNN/blob/main/Bilder/bauer2.png?raw=true', 
    'Gesund': 'https://github.com/schmidleonard/Kartoffel_CNN/blob/main/Bilder/bauer1.png?raw=true'
}

RESULT_DESCRIPTION_MAP = {
    'Early Blight': '⚠️Bei Early Blight entfernen Sie die Blätter mit den "Schießscheiben"-Flecken und spritzen die Pflanze anschließend mit einem Fungizid, um die gesunden Blätter vor einer Neuinfektion zu schützen.',
    'Late Blight': '❗⚠️❗Bei Late Blight müssen Sie sofort alle befallenen Teile radikal entfernen und die Pflanze umgehend mit einem speziellen, wirksamen Fungizid spritzen, um die aggressive Ausbreitung zu stoppen und die Knollen zu retten.',
    'Gesund': '✅Die Pflanze ist gesund. Keine Maßnahmen erforderlich.'
}

# --- Hilfsfunktionen ---
def process_image(image_pil):
    """Bereitet ein PIL-Bild vor und führt die Vorhersage durch."""
    if model is None:
        return None, 0.0

    st.write("Verarbeite Bild...")
    
    # Bild für das Modell vorbereiten
    image = image_pil.resize((224, 224)) # Bildgröße für das Modell
    image_array = np.array(image)
    
    # Falls das Bild 4 Kanäle hat (RGBA), auf 3 Kanäle (RGB) reduzieren
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
        
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Vorhersage
    try:
        prediction = model.predict(image_array)
        
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(prediction)
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Fehler bei der Vorhersage: {e}")
        return None, 0.0

def reset_app_state():
    """Setzt den Zustand zurück, um eine neue Aufnahme zu ermöglichen."""
    st.session_state.result = None

# --- Initialisierung des Session State ---
if 'result' not in st.session_state:
    st.session_state.result = None


st.title("Pflanzenkrankheits-Klassifizierung 🪴")

# Überprüfe, ob das Modell geladen wurde
if model is None:
    st.stop()

# ZUSTAND 1: Ergebnis anzeigen
# -----------------------------
if st.session_state.result:
    predicted_class, confidence = st.session_state.result
    
    st.header("Ergebnis der Klassifizierung")
    
# --- CSS-Anpassung für die Bildhöhe ---
    st.markdown(
        """
        <style>
        /* Zielt auf das Bild-Element innerhalb der st.image Komponente */
        div[data-testid="stImage"] img {
            max-height: 450px;   /* <- HIER die maximale Höhe anpassen */
            object-fit: contain; /* Stellt sicher, dass das Seitenverhältnis erhalten bleibt */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Zeige das Ergebnisbild
    result_img_url = RESULT_IMAGE_MAP.get(predicted_class, "https://placehold.co/600x400/ccc/000?text=Unbekannt")
    result_info_text = RESULT_DESCRIPTION_MAP.get(predicted_class, "Keine Beschreibung verfügbar.")
    col1, col2 = st.columns(2)
    with col1:
        st.image(result_img_url, caption=f"Ergebnis: {predicted_class}", use_container_width=True)
    
    with col2:
        st.success(f"**Vorhersage:** {predicted_class}")
        #st.write(f"**Konfidenz:** {confidence*100:.2f}%")
        
        if confidence < 0.75:
            st.warning("Die Konfidenz ist relativ niedrig. Das Ergebnis könnte unsicher sein.")
            
        st.info(result_info_text)

    st.divider()
    
    # Button zum Zurücksetzen und Starten einer neuen Aufnahme
    st.button("Neues Bild scannen", on_click=reset_app_state, type="primary", use_container_width=True)

# ZUSTAND 2: Eingabe anzeigen
# -----------------------------
else:
    st.header("Neues Bild aufnehmen oder hochladen")
    st.info("Bitte wähle eine der beiden Optionen, um ein Bild zu klassifizieren.")

    image_to_process = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Option 1: Bild hochladen")
        uploaded_file = st.file_uploader(
            "Wähle ein Bild...", 
            type=["jpg", "png", "jpeg"], 
            key="uploaded_file"
        )
        if uploaded_file:
            try:
                image_to_process = Image.open(uploaded_file)
                st.write(f"Datei '{uploaded_file.name}' hochgeladen.")
            except Exception as e:
                st.error(f"Konnte Bilddatei nicht öffnen: {e}")

    with col2:
        st.subheader("Option 2: Foto aufnehmen")
        picture = st.camera_input(
            label="Kamera", 
            help="Kamera erlauben. Auf Mobilgeräten wird oft die Frontkamera bevorzugt.",
            key="picture"
        )
        if picture:
            try:
                image_to_process = Image.open(picture)
                st.write("Foto erfolgreich aufgenommen.")
            except Exception as e:
                st.error(f"Konnte Kamerabild nicht verarbeiten: {e}")

    # Verarbeitung auslösen, wenn ein Bild vorhanden ist
    if image_to_process:
        with st.spinner("Bild wird analysiert..."):
            predicted_class, confidence = process_image(image_to_process)
            
            if predicted_class is not None:
                st.session_state.result = (predicted_class, confidence)
                st.rerun()
            else:
                st.error("Die Analyse konnte nicht abgeschlossen werden.")

