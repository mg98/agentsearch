import os
import faiss
import json
import numpy as np
from langchain_core.documents import Document
from agentsearch.utils.globals import THRESHOLD

FAISS_DIR = "faiss"

_index_cache = {}

def load_index(collection_name: str):
    """Load vector index and metadata for a collection (cached)"""
    if collection_name in _index_cache:
        return _index_cache[collection_name]

    index_path = os.path.join(FAISS_DIR, f"{collection_name}.bin")
    meta_path = os.path.join(FAISS_DIR, f"{collection_name}_meta.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(index_path)

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    _index_cache[collection_name] = (index, metadata)
    return index, metadata

def search_collection(
    collection_name: str,
    query_embedding: np.ndarray,
    k: int = 100,
    threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search a vector collection with a query embedding.

    Returns:
        distances: Array of distances for matched items
        indices: Array of indices for matched items
    """
    index, _ = load_index(collection_name)
    query_vector = query_embedding.reshape(1, -1).astype('float32')

    distances, indices = index.search(query_vector, k)

    if threshold is not None:
        mask = distances[0] < threshold
        distances = distances[0][mask]
        indices = indices[0][mask]
    else:
        distances = distances[0]
        indices = indices[0]

    return distances, indices

def retrieve_documents(
    collection_name: str,
    query_embedding: np.ndarray,
    k: int = 100,
    threshold: float = THRESHOLD
) -> list[Document]:
    """
    Retrieve documents from a vector collection based on similarity.

    Args:
        collection_name: Name of the FAISS collection
        query_embedding: Query embedding vector
        k: Number of results to retrieve
        threshold: Distance threshold for filtering results

    Returns:
        List of Document objects
    """
    index, metadata = load_index(collection_name)
    query_vector = query_embedding.reshape(1, -1).astype('float32')

    distances, indices = index.search(query_vector, k)

    documents = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        distance = distances[0][i]

        if distance < threshold:
            doc_content = metadata['documents'][idx]
            doc_metadata = metadata['metadatas'][idx] if metadata['metadatas'][idx] is not None else {}
            documents.append(Document(page_content=doc_content, metadata=doc_metadata))

    return documents

_questions_embeddings = None
_questions_id_to_idx = None

def load_questions_embeddings():
    """Load pre-computed question embeddings (lazy-loaded singleton)"""
    global _questions_embeddings, _questions_id_to_idx

    if _questions_embeddings is None:
        meta_path = os.path.join(FAISS_DIR, "questions_meta.json")
        embeddings_path = os.path.join(FAISS_DIR, "questions_embeddings.npy")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Questions metadata not found at {meta_path}")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Questions embeddings not found at {embeddings_path}")

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        _questions_id_to_idx = {int(qid): idx for idx, qid in enumerate(metadata['ids'])}
        _questions_embeddings = np.load(embeddings_path)

    return _questions_embeddings, _questions_id_to_idx

def get_question_embedding(question_id: int) -> np.ndarray:
    """Get embedding for a specific question ID"""
    embeddings_array, id_to_idx = load_questions_embeddings()

    if question_id not in id_to_idx:
        raise ValueError(f"Question ID {question_id} not found in embeddings cache")

    idx = id_to_idx[question_id]
    return embeddings_array[idx]

_agent_embeddings_cache = {}

def load_agent_embeddings(collection: str = "agents"):
    """Load pre-computed agent embeddings for a collection (lazy-loaded singleton)"""
    global _agent_embeddings_cache

    if collection not in _agent_embeddings_cache:
        embeddings_path = os.path.join(FAISS_DIR, f"{collection}_embeddings.npy")
        _, metadata = load_index(collection)

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Agent embeddings file not found at {embeddings_path}")

        agent_id_to_idx = {int(agent_id): idx for idx, agent_id in enumerate(metadata['ids'])}
        embeddings_array = np.load(embeddings_path)

        _agent_embeddings_cache[collection] = (embeddings_array, agent_id_to_idx)

    return _agent_embeddings_cache[collection]

def get_agent_embedding(agent_id: int, collection: str = "agents") -> np.ndarray:
    """Get embedding for a specific agent ID"""
    embeddings_array, id_to_idx = load_agent_embeddings(collection)

    if agent_id not in id_to_idx:
        raise ValueError(f"Agent ID {agent_id} not found in collection {collection}")

    idx = id_to_idx[agent_id]
    return embeddings_array[idx]

def create_index(
    collection_name: str,
    vectors: np.ndarray,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict] | None = None,
    save_embeddings: bool = False
) -> None:
    """
    Create and save a vector index with metadata.

    Args:
        collection_name: Name for the collection (without .bin extension)
        vectors: Numpy array of embeddings (n_samples, embedding_dim)
        ids: List of string IDs for each vector
        documents: List of document texts
        metadatas: Optional list of metadata dictionaries
        save_embeddings: If True, also save embeddings as .npy file
    """
    os.makedirs(FAISS_DIR, exist_ok=True)

    vectors = vectors.astype('float32')
    dim = vectors.shape[1]

    if len(vectors) > 50_000:
        nlist = min(100, len(vectors) // 100)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(vectors)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(vectors)

    index_path = os.path.join(FAISS_DIR, f"{collection_name}.bin")
    faiss.write_index(index, index_path)

    if save_embeddings:
        embeddings_path = os.path.join(FAISS_DIR, f"{collection_name}_embeddings.npy")
        np.save(embeddings_path, vectors)

    meta_path = os.path.join(FAISS_DIR, f"{collection_name}_meta.json")
    metadata = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas if metadatas is not None else [None] * len(ids)
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
