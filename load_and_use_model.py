from tensorflow import keras
import json
import numpy as np
from tensorflow.keras.preprocessing import image

# Pfade anpassen!
MODEL_PATH = '/content/potato_disease_model.keras'  # Pfad zum gespeicherten Modell
CLASS_NAMES_PATH = '/content/class_names.json'     # Pfad zu class_names.json
TEST_IMAGE_PATH = '/bestimmt.jpeg'  # Pfad zum Bild


IMG_SIZE = 224  # Muss mit Trainingsgröße übereinstimmen

# Modell laden
model = keras.models.load_model(MODEL_PATH)

# Klassennamen laden
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

def predict_potato_disease(image_path, model, class_names):
    # Bild laden und vorbereiten
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Vorhersage
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    
    predicted_class = class_names[str(predicted_class_idx)]
    
    return predicted_class, confidence

# Vorhersage machen und Ergebnis ausgeben
predicted_class, confidence = predict_potato_disease(TEST_IMAGE_PATH, model, class_names)
print(f"Vorhergesagte Klasse: {predicted_class}")
print(f"Konfidenz: {confidence:.2f}")
