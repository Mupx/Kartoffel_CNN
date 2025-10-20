import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = kagglehub.dataset_download("faysalmiah1721758/potato-dataset")
print(f"Dataset Pfad: {path}")

dataset_path = Path(path)
for item in dataset_path.rglob("*"):
    if item.is_dir():
        print(f"Ordner Name: {item.name}")
        image_count = len(list(item.glob("*.png"))) + len(list(item.glob("*.jpg"))) + len(list(item.glob("*.jpeg")))
        if image_count > 0:
            print(f"{image_count} Bilder gefunden")
        else:
            print("Keine Bilder gefunden")

IMG_SIZE = 224 
BATCH_SIZE = 32 

train_dir = dataset_path #Trainingsordner

print(f"Trainingsorder ist {train_dir}")
print(f"Bildgroesse ist {IMG_SIZE} X {IMG_SIZE}")
print(f"Es werden {BATCH_SIZE} Bilder pro Batch (auf einmal) betrachtet")


#Trainingsdaten modifizieren
train_datagen = ImageDataGenerator(
    rescale=1/255, #RGB Werte werden normiert
    rotation_range=20, #Max 20° rotation
    width_shift_range=0.2, #Verschiebung links-recht (Relativ zur Breite)
    height_shift_range=0.2, #Verschiebung unten-oben (Relativ zur Höhe)
    shear_range=0.2, #Verzerrung
    zoom_range=0.2,#Zufälliger Zoom 0,8 bis 1,2
    horizontal_flip=True,
    validation_split=0.2, #20% werden zur Validierung verwendet    
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"Gefundene Klassen: {train_generator.class_indices}")
num_classes = len(train_generator.class_indices)



#Das eigentliche CNN

def create_small_cnn(num_classes):
    model = keras.Sequential([
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_small_cnn(num_classes)
model.summary()

# def save_feature_maps_comparison(model, img_array, save_dir='feature_maps'):
#     """
#     Speichert Original-Bild und jeweils 3 Feature Maps pro Conv-Block
#     """
#     # Erstelle Ausgabeordner
#     Path(save_dir).mkdir(exist_ok=True)
    
#     # Alle Conv2D Layer finden
#     conv_layer_names = [layer.name for layer in model.layers 
#                         if 'conv' in layer.name.lower()]
    
#     # Feature Map Modell erstellen
#     layer_outputs = [model.get_layer(name).output for name in conv_layer_names]
#     feature_map_model = keras.Model(
#         inputs=model.layers[0].input, 
#         outputs=layer_outputs
#     )
    
#     # Feature Maps berechnen
#     feature_maps = feature_map_model.predict(img_array, verbose=0)
    
#     print(f"\nSpeichere Feature Maps in Ordner: {save_dir}/\n")
    
#     # Für jeden Conv-Block: Original + 3 Feature Maps speichern
#     for block_idx, (layer_name, feature_map) in enumerate(zip(conv_layer_names, feature_maps), 1):
#         num_features = feature_map.shape[-1]
        
#         # Figure mit 4 Subplots erstellen
#         fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#         fig.suptitle(f'Block {block_idx}: {layer_name} ({num_features} Feature Maps gesamt)', 
#                      fontsize=16, fontweight='bold')
        
#         # Original-Bild
#         axes[0].imshow(img_array[0])
#         axes[0].set_title('Original-Bild', fontsize=12)
#         axes[0].axis('off')
        
#         # 3 Feature Maps zeigen
#         feature_indices = [0, num_features//2, num_features-1]  # Erste, mittlere, letzte
        
#         for i, feat_idx in enumerate(feature_indices, 1):
#             if feat_idx < num_features:
#                 channel_image = feature_map[0, :, :, feat_idx]
                
#                 # Werte für Visualisierung
#                 vmin, vmax = channel_image.min(), channel_image.max()
                
#                 im = axes[i].imshow(channel_image, cmap='viridis')
#                 axes[i].set_title(f'Feature Map {feat_idx}\n'
#                                 f'Range: [{vmin:.2f}, {vmax:.2f}]', 
#                                 fontsize=10)
#                 axes[i].axis('off')
                
#                 # Colorbar hinzufügen
#                 plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
#         plt.tight_layout()
        
#         # Speichern
#         filename = f'{save_dir}/block_{block_idx}_{layer_name}.png'
#         plt.savefig(filename, dpi=150, bbox_inches='tight')
#         plt.close()  # Figure schließen um Speicher zu sparen
        
#         print(f"[OK] Gespeichert: {filename}")
        
#         # Info ausgeben
#         print(f"  Output Shape: {feature_map.shape}")
#         print(f"  Anzahl Feature Maps: {num_features}")
#         print(f"  Spatial Groesse: {feature_map.shape[1]} x {feature_map.shape[2]}")
#         print(f"  Wertebereich: [{feature_map.min():.3f}, {feature_map.max():.3f}]\n")
    
#     print(f"\n{'='*60}")
#     print(f"Alle Feature Maps wurden erfolgreich gespeichert!")
#     print(f"Ordner: {save_dir}/")
#     print(f"{'='*60}\n")


# # Ein Testbild holen
# test_images, test_labels = next(train_generator)
# test_img = test_images[0:1]  # Erstes Bild nehmen

# print("\n" + "="*60)
# print("FEATURE MAPS VISUALISIERUNG - Speichere Bilder...")
# print("="*60 + "\n")

# # Feature Maps speichern
# save_feature_maps_comparison(model, test_img, save_dir='feature_maps')

# print("""
# ERKLAERUNG DER GESPEICHERTEN BILDER:

# Block 1 (32 Maps): 
#   - Einfache Muster wie Kanten, Farbverlaeufe
#   - Hellere Bereiche = staerkere Aktivierung
#   - Jede Map reagiert auf unterschiedliche Muster

# Block 2 (64 Maps):
#   - Kombiniert die 32 Maps von Block 1
#   - Erkennt komplexere Formen (z.B. Flecken-Raender)
#   - Kleinere Aufloesung durch MaxPooling

# Block 3 & 4:
#   - Noch abstraktere Muster
#   - Erkennen Kombinationen von Block 2
#   - Sehr kleine Aufloesung, aber semantisch reich

# Die Grauwerte sind KONTINUIERLICH, nicht binaer!
# Das ermoeglicht nuancierte Informationen fuer spaetere Blocks.

# Schau dir die Bilder im Ordner 'feature_maps' an!
# """)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    # keras.callbacks.ModelCheckpoint(
    #     "best_potato_model.keras",
    #     monitor="val_accuracy",
    #     save_best_only=True,
    #     save_format='h5'
    # )

]

EPOCHS = 100

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

#Visualisieren
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy Graph
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss Graph
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)

model.save('potato_disease_model.keras')

# Klassennamen speichern
import json
class_names = {v: k for k, v in train_generator.class_indices.items()}
with open("class_names.json", "w") as f:
    json.dump(class_names, f)







