from langchain_ollama import OllamaEmbeddings
import os

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

EMBEDDING_DIM = 768
