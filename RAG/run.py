import sys
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

try:
    from langchain.chains import RetrievalQA
except ImportError:
    try:
        from langchain_classic.chains import RetrievalQA
    except ImportError:
        print("ERROR RetrievalQA'.")
        sys.exit(1)

CARPETA_CHROMA = "./chroma_db_proyecto"
URL_LM_STUDIO = "http://localhost:1234/v1" 

if not os.path.exists(CARPETA_CHROMA):
    print(f"Error '{CARPETA_CHROMA}'.")
    sys.exit()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("Conectando conchroma...")
vector_db = Chroma(
    persist_directory=CARPETA_CHROMA,
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

print(f"Conectando con LM Studio: {URL_LM_STUDIO}...")
llm = ChatOpenAI(
    base_url=URL_LM_STUDIO,
    api_key="lm-studio",
    temperature=0.3,
    model="local-model"
)

template_filosofico = """
Eres un asistente de investigación experto en filosofía.
Utiliza el siguiente contexto para responder a la pregunta.

CONTEXTO:
{context}

PREGUNTA: 
{question}

RESPUESTA DETALLADA:
"""

prompt = PromptTemplate(
    template=template_filosofico,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

print("\nEscribe 'bye' para salir.")
print("="*60)

while True:
    pregunta = input("\nHaz tu pregunta: ")
    if pregunta.lower() in ['bye']:
        break
    
    try:
        print("Consultando ...")
        resultado = qa_chain.invoke({"query": pregunta})
        print("\nRESPUESTA:")
        print(resultado["result"])
        print("\nFUENTES:")
        for doc in resultado["source_documents"]:
            print(f" - {doc.metadata.get('source', 'Desconocido')}")
    except Exception as e:
        print(f"Error: {e}")