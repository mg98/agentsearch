from langchain_ollama import OllamaEmbeddings

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

EMBEDDING_DIM = 768