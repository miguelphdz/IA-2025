import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_reddit_search_with_comments(search_query, post_limit=10, comment_limit=5):
    """
    Scrapea los resultados de búsqueda de Reddit, extrayendo el título
    del post y los 5 comentarios principales de cada uno.

    Argumentos:
    search_query (str): El tema a buscar.
    post_limit (int): Número de posts a procesar.
    comment_limit (int): Número de comentarios a extraer por post.
    """
    
    base_url = "https://old.reddit.com/search"
    params = {
        'q': search_query,
        'sort': 'relevance'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    dataset = [] # Aquí irán los comentarios
    posts_para_visitar = [] # Aquí guardaremos los enlaces de la Fase 1

    print(f"--- Iniciando scraping de búsqueda: '{search_query}' ---")
    print(f"Objetivo: {post_limit} posts y {comment_limit} comentarios de cada uno.")

    
    try:
        print(f"\nFASE 1: Accediendo a la página de búsqueda...")
        list_response = requests.get(base_url, params=params, headers=headers)
        print(f"URL de búsqueda: {list_response.url}")
        list_response.raise_for_status()
        list_soup = BeautifulSoup(list_response.text, 'html.parser')

        # Buscamos los contenedores de resultados
        posts_divs = list_soup.find_all('div', class_='search-result')

        for post_div in posts_divs:
            if len(posts_para_visitar) >= post_limit:
                break
                
            title_tag = post_div.find('a', class_='search-title')
            comments_tag = post_div.find('a', class_='search-comments')
            
            if title_tag and comments_tag:
                post_titulo = title_tag.text
                post_url_comentarios = comments_tag['href']
                
                posts_para_visitar.append({
                    "titulo": post_titulo,
                    "url": post_url_comentarios
                })
        
        print(f"FASE 1: Completada. Se encontraron {len(posts_para_visitar)} posts para visitar.")

    except requests.exceptions.HTTPError as err:
        print(f"Error de HTTP en FASE 1: {err}")
        return pd.DataFrame() # Devuelve un DataFrame vacío
    except requests.exceptions.RequestException as e:
        print(f"Error de Conexión en FASE 1: {e}")
        return pd.DataFrame()

    print("\nFASE 2: Visitando posts y extrayendo comentarios...")
    
    for i, post in enumerate(posts_para_visitar, 1):
        post_titulo = post['titulo']
        post_url = post['url']
        
        print(f"  -> Visitando Post {i}/{len(posts_para_visitar)}: {post_titulo[:50]}...")
        
        try:
            time.sleep(2) # 2 segundos entre cada post
            
            post_response = requests.get(post_url, headers=headers)
            post_response.raise_for_status()
            post_soup = BeautifulSoup(post_response.text, 'html.parser')
            
            # Buscar los contenedores de comentarios
            comentarios_divs = post_soup.find_all('div', class_='comment', limit=comment_limit)
            
            if not comentarios_divs:
                print("      -> No se encontraron comentarios en este post.")
                continue

            comentarios_encontrados_en_post = 0
            for comment_div in comentarios_divs:
                autor_tag = comment_div.find('a', class_='author')
                texto_tag = comment_div.find('div', class_='md')
                
                if autor_tag and texto_tag:
                    autor_comentario = autor_tag.text
                    texto_comentario = texto_tag.get_text(separator='\n', strip=True)
                    
                    # Guardamos una fila por cada comentario
                    dataset.append({
                        "post_titulo": post_titulo,
                        "comentario_autor": autor_comentario,
                        "comentario_texto": texto_comentario
                    })
                    comentarios_encontrados_en_post += 1
            
            print(f"      -> Se extrajeron {comentarios_encontrados_en_post} comentarios.")

        except requests.exceptions.HTTPError as err:
            print(f"      -> Error HTTP visitando post: {err}. Saltando al siguiente.")
        except requests.exceptions.RequestException as e:
            print(f"      -> Error de Conexión visitando post: {e}. Saltando al siguiente.")

    print(f"\n--- Scraping finalizado. Total de comentarios extraídos: {len(dataset)} ---")
    
    df = pd.DataFrame(dataset)
    return df

# --- Ejecución del script ---

tema_de_busqueda = "carlos manzo" 
posts_a_extraer = 20
comentarios_por_post = 10

mi_dataset = scrape_reddit_search_with_comments(tema_de_busqueda, posts_a_extraer, comentarios_por_post)

print("\n--- Vista previa del Dataset ---")
print(mi_dataset.head())


# --- MODIFICACIÓN PARA GUARDAR EN TXT ---

# 1. Define el nombre del archivo .txt
txt_filename = f"{tema_de_busqueda.replace(' ', '_')}_con_comentarios.txt"

try:
    # 2. Abre el archivo en modo escritura ('w') con codificación utf-8
    with open(txt_filename, 'w', encoding='utf-8') as f:
        
        # 3. Itera sobre cada fila del DataFrame 'mi_dataset'
        for index, row in mi_dataset.iterrows():
            
            # 4. Escribe los datos en un formato legible
            f.write("==================================================\n")
            f.write(f"POST TÍTULO: {row['post_titulo']}\n")
            f.write("--------------------------------------------------\n")
            f.write(f"AUTOR COMENTARIO: {row['comentario_autor']}\n")
            f.write("TEXTO COMENTARIO:\n")
            f.write(row['comentario_texto'])
            f.write("\n\n") # Dos saltos de línea para separar entradas

    print(f"\nDataset guardado como '{txt_filename}'")

except Exception as e:
    print(f"\nOcurrió un error al guardar el archivo .txt: {e}")