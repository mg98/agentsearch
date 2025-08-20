import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agentsearch.utils.globals import db_location, embeddings

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