import sys
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from agentsearch.dataset.questions import questions_store, questions_df
from agentsearch.dataset.agents import agents_store, agents_df
from agentsearch.dataset.agents import Agent
from agentsearch.utils.globals import db_location, embeddings

def create_paper_collection(agent: Agent):
    vector_store = Chroma(
        collection_name=f'agent_{agent.id}',
        persist_directory=db_location,
        embedding_function=embeddings
    )
    vector_store.reset_collection()

    for paper in agent.papers:
        chunks = paper.make_chunks()
        if len(chunks) == 0:
            print(f"No chunks found for {paper.id}")
            continue
        print(f"Adding {len(chunks)} chunks from {paper.id}")

        # Process chunks in batches
        BATCH_SIZE = 32
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            batch_ids = [f"{paper.id}_{j}" for j in range(i, i + len(batch_chunks))]
            vector_store.add_documents(
                documents=batch_chunks,
                ids=batch_ids
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
    agents = Agent.all(shallow=True)

    documents = [Document(
        page_content=", ".join(agent.research_fields),
        metadata={
            "name": agent.name,
            "scholar_url": agent.scholar_url
        }) for agent in agents]
    
    agents_store.add_documents(
        documents=documents,
        ids=[str(agent.id) for agent in agents]
    )

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['questions', 'agents', 'papers']:
        print("Usage: python -m scripts.embed [questions|agents|papers]")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    # Create requested collection
    if mode == 'questions':
        questions_store.reset_collection()
        print("Creating question collection...")
        create_question_collection()
    elif mode == 'agents':
        agents_store.reset_collection()
        print("Creating agents collection...")
        create_agent_collection()
    elif mode == 'papers':  # papers mode
        print("Creating paper collections...")
        for agent in tqdm(Agent.all(shallow=True), desc="Agents"):
            agent.load_papers()
            create_paper_collection(agent)
    else:
        print("Invalid mode")
        sys.exit(1)
    
    print(f"Successfully created {mode} collection")

    
    # response = input("This will delete chroma_db, are you sure you want to proceed? (y/n): ")
    # if response.lower() != 'y':
    #     print("Aborting...")
    #     exit()
    
    # shutil.rmtree(db_location, ignore_errors=True)

    # from agentsearch.dataset.agents import agents_store, agents_df
    # from agentsearch.dataset.questions import questions_store, questions_df

    # # Create question collection
    # print("Creating question collection...")
    # create_question_collection()

    # # Create question collection
    # print("Creating authors collection...")
    # create_agent_collection()

    # # Create paper collection
    # print("Creating paper collection...")
    # for id, _ in agents_df.iterrows():
    #     create_paper_collection(id)

   