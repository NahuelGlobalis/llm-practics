"""
Este módulo contiene funciones para crear el retriever de reseñas de pacientes.

Un retriever en LangChain es un componente que permite buscar y recuperar 
información relevante desde una fuente de datos. Es una parte fundamental 
de los sistemas RAG (Retrieval-Augmented Generation).
"""

import dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Definimos las rutas para acceder a los datos y almacenar los embeddings
REVIEWS_CSV_PATH = "data/reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

# Cargamos variables de entorno (buena práctica para manejar claves API, etc.)
dotenv.load_dotenv()

# PASO 1: Cargamos los documentos
# Los Document Loaders en LangChain permiten importar datos de diferentes fuentes
# CSVLoader específicamente permite cargar datos desde archivos CSV
loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="review")
reviews = loader.load()

# PASO 2: Creamos la base de datos vectorial con Chroma
# Chroma es un vector store que permite indexar documentos mediante embeddings
# y realizar búsquedas semánticas eficientes
reviews_vector_db = Chroma.from_documents(
    # Documentos a indexar
    documents=reviews,
    # Modelo de embeddings que transformará el texto en vectores numéricos
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    # Directorio donde se persistirán los embeddings para uso futuro
    persist_directory=REVIEWS_CHROMA_PATH
)

# PASO 3: Guardamos la base de datos vectorial en disco
# Esto permite reutilizarla sin necesidad de recrear los embeddings cada vez
reviews_vector_db.persist()

# Nota: Para usar este retriever en una aplicación, se accede así:
# retriever = reviews_vector_db.as_retriever(k=10)
# Donde k es el número de documentos similares a recuperar
