import shutil
import shutil
import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter
from io import StringIO
from typing import List
import pandas as pd

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if __name__ == "__main__":
    response = input("This will delete chroma_db, are you sure you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborting...")
        exit()
    
    shutil.rmtree(db_location, ignore_errors=True)

    from agentsearch.dataset.agents import agents_store, agents_df
    from agentsearch.dataset.questions import questions_store, questions_df


    def create_chunks(path_to_html_doc: str) -> List[Document]:
        headers_to_split_on = [
            ("h1", "title"),
            ("h2", "section"),
            ("h3", "subsection"),
            ("h4", "subsubsection"),
            ("h6", "abstract"),
            ("p", "paragraph"),
        ]
        html_splitter = HTMLSectionSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_element=True)
        
        with open(path_to_html_doc, 'r') as f:
            file_content = f.read()
        file = StringIO(file_content)
        chunks = html_splitter.split_text_from_file(file)
        chunks = [chunk for chunk in chunks if chunk.metadata.get('paragraph') and len(chunk.page_content) > 100]
        return chunks

    def create_paper_collection(agent_id: int):
        vector_store = Chroma(
            collection_name=f'agent_{agent_id}',
            persist_directory=db_location,
            embedding_function=embeddings
        )

        html_files = glob.glob(f"papers/html/{agent_id}/*.html")
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

    def create_question_collection():
        documents = [Document(
            page_content=row['question'],
            metadata={
                "agent_id": row['agent_id']
            }) for _, row in questions_df.iterrows()]
        
        questions_store.add_documents(
            documents=documents,
            ids=[str(i) for i in questions_df.index.tolist()]
        )

    def create_agent_collection():
        documents = [Document(
            page_content=", ".join(row['research_fields']),
            metadata={
                "name": row['name'],
                "scholar_url": row['scholar_url']
            }) for _, row in agents_df.iterrows()]
        
        agents_store.add_documents(
            documents=documents,
            ids=[str(i) for i in agents_df.index.tolist()]
        )

    # Create question collection
    print("Creating question collection...")
    create_question_collection()

    # Create question collection
    print("Creating authors collection...")
    create_agent_collection()

    # Create paper collection
    print("Creating paper collection...")
    for id, _ in agents_df.iterrows():
        create_paper_collection(id)

   