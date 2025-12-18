import os
import glob
import random
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, 
    load_img, 
    img_to_array, 
    array_to_img
)

SOURCE_DIR = "dataset_final/morada"
CLASS_PREFIX = "morada" 
TARGET_COUNT = 2000      
IMAGE_SIZE = (224, 224)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("Iniciando proceso de aumento de datos...")

print(f"Cargando imágenes originales de {SOURCE_DIR}...")


original_image_paths = glob.glob(os.path.join(SOURCE_DIR, "*.*"))
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
original_image_paths = [p for p in original_image_paths if p.lower().endswith(image_extensions)]

if not original_image_paths:
    print(f"[ERROR].")
    sys.exit()

original_images = []
for path in original_image_paths:
    try:
        img = load_img(path, target_size=IMAGE_SIZE)
        original_images.append(img_to_array(img))
    except Exception as e:
        print(f"\n[!] Error al cargar o redimensionar {os.path.basename(path)}: {e}")
        continue


current_count = len(original_images)
print(f"Se cargaron {current_count} imágenes originales.")

new_file_counter = current_count
images_needed = TARGET_COUNT - current_count

if images_needed <= 0:
    print(f"Ya tienes {current_count} imágenes o más. No se necesita aumento.")
    sys.exit()

print(f"Se generarán {images_needed} imágenes nuevas...")

while new_file_counter < TARGET_COUNT:
    img_array = random.choice(original_images)
    
    img_array = img_array.reshape((1,) + img_array.shape)
    
    augmented_image_iter = datagen.flow(img_array, batch_size=1)
    augmented_image_array = next(augmented_image_iter)[0].astype('uint8')
    
    save_path = os.path.join(SOURCE_DIR, f"{CLASS_PREFIX}_{new_file_counter:05d}.jpg")
    
    img_pil = array_to_img(augmented_image_array)
    img_pil.save(save_path, "JPEG")
    
    new_file_counter += 1
    
    sys.stdout.write(f"\rImágenes generadas: {new_file_counter}/{TARGET_COUNT}")
    sys.stdout.flush()

print(f"\n\n--- ¡Proceso completado! ---")
print(f"La carpeta '{SOURCE_DIR}' ahora contiene {new_file_counter} imágenes.")