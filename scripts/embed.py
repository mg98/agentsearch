import sys
import torch
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agentsearch.dataset.questions import questions_store, questions_df
from agentsearch.dataset.agents import Agent, AgentStore
from agentsearch.utils.globals import db_location, embeddings

def create_paper_collection(agent: Agent):
    collection_name = f'agent_{agent.id}'
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )

    try:
        existing_count = vector_store._collection.count()
        if existing_count > 0:
            print(f"Collection {collection_name} already exists with {existing_count} documents, skipping")
            return
    except:
        pass
    
    vector_store.reset_collection()
    agent.load_papers()

    for paper in agent.papers:
        chunks = paper.make_chunks()
        if len(chunks) == 0:
            print(f"No chunks found for {paper.id}")
            continue
        print(f"Adding {len(chunks)} chunks from {paper.id}")

        BATCH_SIZE = 64
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            batch_ids = [f"{paper.id}_{j}" for j in range(i, i + len(batch_chunks))]
            vector_store.add_documents(
                documents=batch_chunks,
                ids=batch_ids
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def create_question_collection():
    documents = [Document(
        page_content=row['question'],
        metadata={
            "agent_id": int(row['agent_id'])
        }) for _, row in questions_df.iterrows()]
    
    BATCH_SIZE = 1024
    ids = [str(i) for i in questions_df.index.tolist()]
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch_documents = documents[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        questions_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )

def create_agent_collection(agent_store: AgentStore):
    agent_store._store.reset_collection()
    agents = agent_store.all(shallow=True)

    documents = [Document(
        page_content=agent.agent_card,
        metadata={
            "name": str(agent.name),
            "scholar_url": str(agent.scholar_url),
            "citation_count": int(agent.citation_count),
            "agent_card": str(agent.agent_card),
        }) for agent in agents]
    
    agent_store._store.add_documents(
        documents=documents,
        ids=[str(agent.id) for agent in agents]
    )

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['questions', 'agents', 'papers']:
        print("Usage: python -m scripts.embed [questions|agents|papers]")
        sys.exit(1)
        
    mode = sys.argv[1]
    if len(sys.argv) > 2:
        job_id = int(sys.argv[2])
        job_count = int(sys.argv[3])
    else:
        job_id = 0
        job_count = 1
    
    if mode == 'questions':
        questions_store.reset_collection()
        print("Creating question collection...")
        create_question_collection()
    elif mode == 'agents':
        # print("Creating agents collection with LLM agent cards...")
        # agent_store = AgentStore(use_llm_agent_card=True)
        # create_agent_collection(agent_store)

        print("Creating agents collection with human agent cards...")
        agent_store = AgentStore(use_llm_agent_card=False)
        create_agent_collection(agent_store)
    elif mode == 'papers':
        print("Creating paper collections...")
        agent_store = AgentStore(use_llm_agent_card=False)
        agents = [agent for i, agent in enumerate(agent_store.all(shallow=True)) if i % job_count == job_id]

        for agent in tqdm(agents, desc="Agents"):
            create_paper_collection(agent)
    else:
        print("Invalid mode")
        sys.exit(1)
    
    print(f"Successfully created {mode} collection")