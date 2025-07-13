from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def retrieve(agent_id: int, query: str, k: int = 20) -> list[Document]:
    vector_store = Chroma(
        collection_name=f"agent_{agent_id}",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": 0.5}
    )

    return retriever.invoke(query)