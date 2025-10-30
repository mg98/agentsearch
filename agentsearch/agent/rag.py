import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agentsearch.utils.globals import db_location, embeddings
import numpy as np
import math

THRESHOLD = 1 #math.sqrt(2) / 2

def retrieve(agent_id: int, query: str, k: int = 100) -> list[Document]:
    vector_store = Chroma(
        collection_name=f"agent_{agent_id}",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5, "k": k}
    )

    # Suppress all output from retriever.invoke
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return retriever.invoke(query)
    
def retrieve_with_embedding(agent_id: int, query_embedding: np.ndarray, k: int = 100) -> list[Document]:
    vector_store = Chroma(
        collection_name=f"agent_{agent_id}",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    search_results = vector_store._collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
    )

    documents = []
    if search_results['documents'] is not None and search_results['documents'][0]:
        for i, doc_content in enumerate(search_results['documents'][0]):
            distance = search_results['distances'][0][i]

            if distance < THRESHOLD:
                metadata = search_results['metadatas'][0][i] if search_results['metadatas'] and search_results['metadatas'][0][i] is not None else {}
                documents.append(Document(page_content=doc_content, metadata=metadata))

    return documents