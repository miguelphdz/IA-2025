import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dirname = 'dataset_procesado' 
IMG_height = 128
IMG_width = 128
BATCH_SIZE = 32
EPOCHS = 20 
MODEL_FILENAME = "modelo_animales_optimizado.h5"

print(f"Buscando imágenes en: {dirname}")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_generator = datagen.flow_from_directory(
    dirname,
    target_size=(IMG_height, IMG_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dirname,
    target_size=(IMG_height, IMG_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_generator.num_classes

try:
    print(f"Buscando archivo '{MODEL_FILENAME}'...")
    model = load_model(MODEL_FILENAME)
    print("Modelo existente cargado exitosamente. Se continuará el entrenamiento.")

except Exception as e:
    print(f"No se encontró modelo previo. Creando nuevo modelo.")
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(IMG_height, IMG_width, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))


model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

print("\n--- Iniciando Entrenamiento --")
print(f"Entrenando por {EPOCHS} épocas adicionales...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

model.save(MODEL_FILENAME)
print(f"Modelo actualizado guardado como '{MODEL_FILENAME}'")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()
