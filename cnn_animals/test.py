import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog

MODEL_PATH = "modelo_animales_optimizado.h5"
IMG_SIZE = (128, 128)

CLASS_NAMES = ['cats', 'dogs', 'hormigas', 'mariquitas', 'tortugas']

# --- 3. CARGAR MODELO ---
print("Cargando el modelo...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

def predecir_imagen():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen para probar",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not file_path:
        print("No seleccionaste ninguna imagen.")
        return False

    print(f"\nProcesando: {os.path.basename(file_path)}")

    try:
        img_original = load_img(file_path) 
        img = load_img(file_path, target_size=IMG_SIZE) 
        
        img_array = img_to_array(img)
        img_array = img_array / 255.0  
        
        
        img_array = np.expand_dims(img_array, axis=0)

        
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0]) 
        
        class_idx = np.argmax(predictions[0])
        class_label = CLASS_NAMES[class_idx]

        plt.figure(figsize=(6, 6))
        plt.imshow(img_original)
        plt.axis('off')
        
        titulo = f"Predicción: {class_label.upper()}"
        plt.title(titulo, color='green', fontsize=14, fontweight='bold')
        plt.show()
        
        return True

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return False

if __name__ == "__main__":
    while True:
        continuar = input("\n¿Quieres probar una imagen? (s/n): ").lower()
        if continuar != 's':
            print("Cerrando programa.")
            break
        
        predecir_imagen()