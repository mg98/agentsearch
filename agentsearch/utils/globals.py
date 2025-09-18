import requests

db_location = "./chroma_db"
EMBEDDING_DIM = 768

def is_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        return response.status_code == 200
    except:
        return False

if is_ollama_available():
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Using Ollama embeddings")
else:
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={'device': 'cuda', 'trust_remote_code': True},  # Use 'cpu' if no GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    print("Using HuggingFace embeddings")
