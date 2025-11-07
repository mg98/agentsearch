import sys
import torch
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import questions_df
from agentsearch.utils.globals import embeddings
from agentsearch.utils.vector_store import create_index
from agentsearch.dataset.registries import get_faculties, get_departments, get_groups

def embed_with_retry(texts: list[str], max_retries: int = 3, base_delay: float = 2.0) -> list:
    """Embed texts with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            return embeddings.embed_documents(texts)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"\n  Retry {attempt + 1}/{max_retries} after {delay}s delay: {e}")
            time.sleep(delay)

def create_question_collection():
    """Create FAISS index for questions with their embeddings"""
    print(f"Computing embeddings for {len(questions_df)} questions...")
    question_ids = questions_df.index.tolist()
    question_texts = questions_df['question'].tolist()

    BATCH_SIZE = 1024
    all_embeddings = []

    for i in tqdm(range(0, len(question_texts), BATCH_SIZE), desc="Embedding questions"):
        batch_texts = question_texts[i:i+BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors = np.array(all_embeddings).astype('float32')

    print(f"Building FAISS index for {len(vectors)} questions, dim={vectors.shape[1]}")

    metadatas = [
        {"agent_id": int(questions_df.loc[qid, 'agent_id']) if pd.notna(questions_df.loc[qid, 'agent_id']) else -1}
        for qid in question_ids
    ]

    create_index(
        collection_name="questions",
        vectors=vectors,
        ids=[str(qid) for qid in question_ids],
        documents=question_texts,
        metadatas=metadatas,
        save_embeddings=True
    )

    print("Successfully created questions FAISS index")

def create_paper_collection(agent: Agent):
    """Create FAISS index for an agent's paper chunks"""
    collection_name = f'agent_{agent.id}'

    index_path = os.path.join("faiss", f"{collection_name}.bin")
    if os.path.exists(index_path):
        print(f"Collection {collection_name} already exists, skipping")
        return

    if len(agent.papers) == 0:
        print(f"No papers found for agent {agent.id}, skipping")
        return

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for paper in agent.papers:
        chunks = paper.make_chunks()
        if len(chunks) == 0:
            print(f"No chunks found for {paper.id}")
            continue
        print(f"Adding {len(chunks)} chunks from {paper.id}")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk.page_content)
            all_ids.append(f"{paper.id}_{i}")
            all_metadatas.append(chunk.metadata if chunk.metadata else {})

    if len(all_chunks) == 0:
        print(f"No chunks to embed for agent {agent.id}, skipping")
        return

    print(f"Computing embeddings for {len(all_chunks)} chunks...")
    all_embeddings = []

    for i, text in enumerate(tqdm(all_chunks, desc=f"Embedding agent {agent.id}")):
        embedding = embed_with_retry([text])
        all_embeddings.extend(embedding)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors = np.array(all_embeddings).astype('float32')

    print(f"Building FAISS index for agent {agent.id}: {len(vectors)} chunks, dim={vectors.shape[1]}")

    create_index(
        collection_name=collection_name,
        vectors=vectors,
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas,
        save_embeddings=False
    )

    print(f"Successfully created FAISS index for agent {agent.id}")

def create_registries_collection():
    """Create FAISS index for registries (faculties, departments, groups)"""
    documents = []
    metadatas = []

    for faculty in get_faculties():
        documents.append(faculty)
        metadatas.append({"parent": "root"})

    for faculty in get_faculties():
        for department in get_departments(faculty):
            documents.append(department)
            metadatas.append({"parent": faculty})

        for department in get_departments(faculty):
            for group in get_groups(faculty, department):
                documents.append(group)
                metadatas.append({"parent": department})

    print(f"Computing embeddings for {len(documents)} registry entries...")
    BATCH_SIZE = 512
    all_embeddings = []

    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding registries"):
        batch_texts = documents[i:i+BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors = np.array(all_embeddings).astype('float32')

    print(f"Building FAISS index for {len(vectors)} registries, dim={vectors.shape[1]}")

    create_index(
        collection_name="registries",
        vectors=vectors,
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        metadatas=metadatas,
        save_embeddings=False
    )

    print("Successfully created registries FAISS index")

def create_agent_collection(collection_name: str = "agents"):
    """Create FAISS index for agent embeddings"""
    print(f"Creating {collection_name} collection...")

    agents = Agent.all(collection=collection_name)

    agent_texts = [agent.agent_card for agent in agents]

    print(f"Computing embeddings for {len(agent_texts)} agents...")
    BATCH_SIZE = 512
    all_embeddings = []

    for i in tqdm(range(0, len(agent_texts), BATCH_SIZE), desc="Embedding agents"):
        batch_texts = agent_texts[i:i+BATCH_SIZE]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vectors = np.array(all_embeddings).astype('float32')

    print(f"Building FAISS index for {len(vectors)} agents, dim={vectors.shape[1]}")

    metadatas = [
        {
            "name": str(agent.name),
            "scholar_url": str(agent.scholar_url),
            "citation_count": int(agent.citation_count),
            "agent_card": str(agent.agent_card),
            "faculty": str(agent.faculty),
            "department": str(agent.department),
            "group": str(agent.group),
        }
        for agent in agents
    ]

    create_index(
        collection_name=collection_name,
        vectors=vectors,
        ids=[str(agent.id) for agent in agents],
        documents=agent_texts,
        metadatas=metadatas,
        save_embeddings=True
    )

    print(f"Successfully created {collection_name} FAISS index")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['questions', 'agents', 'papers', 'registries']:
        print("Usage: python -m scripts.embed [questions|agents|papers|registries]")
        sys.exit(1)

    mode = sys.argv[1]
    if len(sys.argv) > 2:
        job_id = int(sys.argv[2])
        job_count = int(sys.argv[3])
    else:
        job_id = 0
        job_count = 1

    if mode == 'questions':
        print("Creating question FAISS index...")
        create_question_collection()
    elif mode == 'agents':
        print("Creating agents collection...")
        create_agent_collection("agents")

        ## Currently failing because of embedding model input token limit
        # print("Creating agents collection with LLM-generated agent cards...")
        # create_agent_collection("agents_llm")
    elif mode == 'papers':
        print("Creating paper collections...")
        all_agents = Agent.all()
        agents = [agent for i, agent in enumerate(all_agents) if i % job_count == job_id]

        for agent in tqdm(agents, desc="Agents"):
            create_paper_collection(agent)
    elif mode == 'registries':
        print("Creating registries collection...")
        create_registries_collection()
    else:
        print("Invalid mode")
        sys.exit(1)
    
    print(f"Successfully created {mode} collection")