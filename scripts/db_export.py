import pandas as pd
from langchain_chroma import Chroma
from agentsearch.utils.globals import db_location, embeddings
from agentsearch.dataset.agents import agents_df
from agentsearch.dataset.questions import questions_df

def export_questions():
    """Export all questions from ChromaDB to questions.parquet"""
    print("Exporting questions...")
    
    # Initialize questions collection
    questions_store = Chroma(
        collection_name='questions',
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    # Get all question IDs from the dataframe
    question_ids = [str(qid) for qid in questions_df.index]
    
    # Retrieve embeddings from ChromaDB
    result = questions_store._collection.get(
        ids=question_ids,
        include=['embeddings']
    )
    
    # Create a list to store question data
    questions_data = []
    
    for i, qid in enumerate(result['ids']):
        qid_int = int(qid)
        embedding = result['embeddings'][i] if result['embeddings'] and len(result['embeddings']) > i else None
        
        question_record = {
            'id': qid_int,
            'embedding': embedding
        }
        questions_data.append(question_record)
    
    # Create DataFrame and save to parquet
    questions_export_df = pd.DataFrame(questions_data)
    questions_export_df.to_parquet('data/questions.parquet', index=False)
    print(f"Exported {len(questions_export_df)} questions to data/questions.parquet")

def export_agents():
    """Export all agents from ChromaDB to agents.parquet"""
    print("Exporting agents...")
    
    # Initialize only human agent cards collection (ignoring LLM embeddings)
    agents_with_human = Chroma(
        collection_name='agents_with_human_agent_cards',
        persist_directory=db_location,
        embedding_function=embeddings
    )
    
    # Get all agent IDs from the dataframe
    agent_ids = [str(aid) for aid in agents_df.index]
    
    # Get embeddings from human agent cards collection
    result = agents_with_human._collection.get(
        ids=agent_ids,
        include=['embeddings']
    )
    
    # Create a list to store agent data
    agents_data = []
    
    for aid_str in result['ids']:
        aid_int = int(aid_str)
        idx = result['ids'].index(aid_str)
        embedding = result['embeddings'][idx] if result['embeddings'] and len(result['embeddings']) > idx else None
        
        agent_record = {
            'id': aid_int,
            'embedding': embedding
        }
        agents_data.append(agent_record)
    
    # Create DataFrame and save to parquet
    agents_export_df = pd.DataFrame(agents_data)
    agents_export_df.to_parquet('data/agents.parquet', index=False)
    print(f"Exported {len(agents_export_df)} agents to data/agents.parquet")

if __name__ == "__main__":
    export_questions()
    export_agents()
    print("Export completed successfully!")