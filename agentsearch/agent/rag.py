from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


def retrieve(collection_name: str, query: str, k: int = 10) -> list[Document]:
    vector_store = Chroma(
        collection_name=str(collection_name),
        persist_directory=db_location,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)