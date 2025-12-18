import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def limpiar_texto(texto):
    texto = texto.lower()
    tokens = nltk.word_tokenize(texto)
    stop_words = set(stopwords.words('spanish'))
    
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

carpeta_pdfs = "./dataset_rag/"
documentos_procesados = []

print("Procesando ")
for archivo in os.listdir(carpeta_pdfs):
    if archivo.endswith(".pdf"):
        ruta = os.path.join(carpeta_pdfs, archivo)
        loader = PyPDFLoader(ruta)
        docs = loader.load()
        
        for doc in docs:
            doc.page_content = limpiar_texto(doc.page_content)
            documentos_procesados.append(doc)

print(f"Procesadas {len(documentos_procesados)} páginas.")

print("\nChunking")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documentos_procesados)

print(f"Dividido en {len(chunks)} chunks.")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

try:
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db_proyecto"
    )
    print("La base de datos se encuentra en la carpeta: ./chroma_db")
except Exception as e:
    print(f"error al crear el Vector Store: {e}")