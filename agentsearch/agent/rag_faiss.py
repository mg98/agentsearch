import os
import faiss
import json
import numpy as np
from langchain_core.documents import Document
from agentsearch.utils.globals import embeddings, THRESHOLD

FAISS_DIR = "faiss"

def _load_faiss_index(collection_name: str):
    index_path = os.path.join(FAISS_DIR, f"{collection_name}.bin")
    meta_path = os.path.join(FAISS_DIR, f"{collection_name}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(index_path)

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    return index, metadata

def retrieve(agent_id: int, query: str, k: int = 100) -> list[Document]:
    collection_name = f"agent_{agent_id}"
    index, metadata = _load_faiss_index(collection_name)

    query_embedding = embeddings.embed_query(query)
    query_vector = np.array([query_embedding]).astype('float32')

    distances, indices = index.search(query_vector, k)

    documents = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        distance = distances[0][i]

        if distance < THRESHOLD:
            doc_content = metadata['documents'][idx]
            doc_metadata = metadata['metadatas'][idx] if metadata['metadatas'][idx] is not None else {}
            documents.append(Document(page_content=doc_content, metadata=doc_metadata))

    return documents

def retrieve_with_embedding(agent_id: int, query_embedding: np.ndarray, k: int = 100) -> list[Document]:
    collection_name = f"agent_{agent_id}"
    index, metadata = _load_faiss_index(collection_name)

    query_vector = query_embedding.reshape(1, -1).astype('float32')

    distances, indices = index.search(query_vector, k)

    documents = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        distance = distances[0][i]

        if distance < THRESHOLD:
            doc_content = metadata['documents'][idx]
            doc_metadata = metadata['metadatas'][idx] if metadata['metadatas'][idx] is not None else {}
            documents.append(Document(page_content=doc_content, metadata=doc_metadata))

    return documents
