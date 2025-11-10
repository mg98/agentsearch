import numpy as np
from langchain_core.documents import Document
from agentsearch.utils.globals import embeddings
from agentsearch.utils.vector_store import retrieve_documents

def retrieve(agent_id: int, query: str, k: int = 100) -> list[Document]:
    """Retrieve documents from an agent's paper collection using a query string"""
    collection_name = f"agent_{agent_id}"
    query_embedding = np.array(embeddings.embed_query(query))
    return retrieve_documents(collection_name, query_embedding, k)

def retrieve_with_embedding(agent_id: int, query_embedding: np.ndarray, k: int = 100) -> list[Document]:
    """Retrieve documents from an agent's paper collection using a pre-computed embedding"""
    collection_name = f"agent_{agent_id}"
    return retrieve_documents(collection_name, query_embedding, k)
