import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirname = 'dataset_procesado'
IMG_height = 128
IMG_width = 128
BATCH_SIZE = 32

print("Cargando modelo 'modelo_animales_optimizado.h5'...")
try:
    model = tf.keras.models.load_model("modelo_animales_optimizado.h5")
except OSError:
    print("Error: No se encontró el archivo 'modelo_animales_optimizado.h5'.")
    print("Asegúrate de haber ejecutado el entrenamiento primero.")
    exit()

print("Preparando datos de prueba...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

test_generator = datagen.flow_from_directory(
    dirname,
    target_size=(IMG_height, IMG_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


class_names = list(test_generator.class_indices.keys())

print("Generando predicciones para", test_generator.samples, "imágenes...")
Y_pred = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names))


correct_indices = np.where(y_pred_classes == y_true)[0]
incorrect_indices = np.where(y_pred_classes != y_true)[0]


print("\n--- Fin de la evaluación ---")