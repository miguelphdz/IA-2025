import os
import shutil
from PIL import Image, ImageOps
import sys
import glob 


INPUT_DIR = "dataset_final/morada"           

OUTPUT_DIR = "cnn_animals/dataset_procesado/mariquitas"      
TARGET_SIZE = (224, 224)              
EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

def process_single_folder():
    print(f"--- Iniciando estandarización de imágenes ---")
    print(f"Origen: {INPUT_DIR}")
    print(f"Destino: {OUTPUT_DIR}")
    print(f"Tamaño objetivo: {TARGET_SIZE}\n")

    if not os.path.exists(INPUT_DIR):
        print(f"Error: No se encuentra la carpeta de entrada '{INPUT_DIR}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_processed = 0
    total_errors = 0
    total_skipped = 0

    files = os.listdir(INPUT_DIR)

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        
        if os.path.isdir(file_path):
            continue
        if not filename.lower().endswith(EXTENSIONS):
            total_skipped += 1
            continue

        try:
            with Image.open(file_path) as img:
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_fitted = ImageOps.fit(img, TARGET_SIZE, Image.Resampling.LANCZOS)
                
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                save_path = os.path.join(OUTPUT_DIR, new_filename)
                
                img_fitted.save(save_path, "JPEG", quality=90)
                
                total_processed += 1
                
                if total_processed % 10 == 0: 
                    sys.stdout.write(f"\rProcesadas: {total_processed} | Errores: {total_errors}")
                    sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error en archivo: {filename} - {e}")
            total_errors += 1

    print(f"\n\n--- ¡Proceso Completado! ---")
    print(f"Imágenes listas en '{OUTPUT_DIR}': {total_processed}")
    print(f"Imágenes corruptas/rotas ignoradas: {total_errors}")
    print(f"Archivos no-imagen ignorados: {total_skipped}")

if __name__ == "__main__":
    process_single_folder()