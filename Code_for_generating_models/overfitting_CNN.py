import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, regularizers
from sklearn.utils.class_weight import compute_class_weight
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
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
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
def create_overfitted_cnn(num_classes):
    model = keras.Sequential([

        
        layers.Conv2D(64, (3,3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        
        layers.Conv2D(1024, (3,3), padding='same', activation='relu'),
        layers.Conv2D(1024, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),

        
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),  
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = create_overfitted_cnn(num_classes)
model.summary()


# Optimizer & Compile
initial_lr = 0.0005
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=initial_lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Class Weights berechnen
labels = train_generator.classes  
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights_array))



EPOCHS = 30

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights
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
    plt.savefig('training_history_overfitting.png')
    plt.show()

plot_training_history(history)

model.save('potato_disease_model_overfitted.keras')

