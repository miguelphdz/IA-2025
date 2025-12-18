import os
from pypdf import PdfReader


CARPETA_PDFS = "./dataset_rag"     
ARCHIVO_DESTINO = "dataset_para_rag.txt" 

def limpiar_texto_basico(texto):
    """
    Quita saltos de línea excesivos y espacios múltiples para ahorrar tokens.
    """
    if not texto:
        return ""
    texto = " ".join(texto.split())
    return texto

def agregar_pdfs():
    if not os.path.exists(CARPETA_PDFS):
        print(f" Error: La carpeta '{CARPETA_PDFS}' no existe.")
        return
    
    if not os.path.exists(ARCHIVO_DESTINO):
        print(f"Advertencia: '{ARCHIVO_DESTINO}' no existe.")

    count = 0

    with open(ARCHIVO_DESTINO, 'a', encoding='utf-8') as f_out:
        
        archivos = [f for f in os.listdir(CARPETA_PDFS) if f.endswith(".pdf")]
        print(f"Se encontraron {len(archivos)} libros en la carpeta.")

        for archivo in archivos:
            ruta_pdf = os.path.join(CARPETA_PDFS, archivo)
            print(f"   ... Procesando: {archivo}")
            
            try:
                reader = PdfReader(ruta_pdf)
                texto_total_libro = ""
                
                for i, page in enumerate(reader.pages):
                    texto_pagina = page.extract_text()
                    if texto_pagina:
                        texto_total_libro += limpiar_texto_basico(texto_pagina) + "\n"
                
                header = f"\n\n{'='*50}\nDOCUMENTO REFERENCIA: {archivo}\n{'='*50}\n\n"
                
                f_out.write(header)
                f_out.write(texto_total_libro)
                
                count += 1
                
            except Exception as e:
                print(f" Error leyendo {archivo}: {e}")

    print("-" * 50)
    print(f"Se agregaron {count} libros al archivo '{ARCHIVO_DESTINO}'.")


if __name__ == "__main__":
    agregar_pdfs()