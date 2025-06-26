import os
import shutil
import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agentsearch.dataset.chunk import create_chunks

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_db"

def create_collection(author_id: str):
    vector_store = Chroma(
        collection_name=author_id,
        persist_directory=db_location,
        embedding_function=embeddings
    )

    html_files = glob.glob(f"papers/html/{author_id}/*.html")
    for html_file in html_files:
        print(f"Processing {html_file}")
        chunks = create_chunks(html_file)
        if len(chunks) == 0:
            print(f"No chunks found for {html_file}")
            continue
        print(f"Adding {len(chunks)} chunks from {html_file}")
        vector_store.add_documents(
            documents=chunks,
            ids=[f"{html_file}_{i}" for i in range(len(chunks))]
        )

if __name__ == "__main__":
    shutil.rmtree(db_location, ignore_errors=True)

    for author_id in os.listdir("papers/html"):
        if not os.path.isdir(os.path.join("papers/html", author_id)):
            continue
        create_collection(author_id)

   