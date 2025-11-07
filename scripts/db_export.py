import pandas as pd
import numpy as np
from agentsearch.dataset.agents import agents_df
from agentsearch.dataset.questions import questions_df, Question
from tqdm import tqdm

def export_questions():
    """Export all questions with computed embeddings to questions.parquet"""
    print("Exporting questions...")

    questions_data = []
    question_ids = questions_df.index.tolist()

    BATCH_SIZE = 100
    for i in tqdm(range(0, len(question_ids), BATCH_SIZE), desc="Computing embeddings"):
        batch_ids = question_ids[i:i+BATCH_SIZE]
        questions = Question.many(batch_ids)

        for question in questions:
            question_record = {
                'id': question.id,
                'embedding': question.embedding.tolist()
            }
            questions_data.append(question_record)

    questions_export_df = pd.DataFrame(questions_data)
    questions_export_df.to_parquet('data/questions.parquet', index=False)
    print(f"Exported {len(questions_export_df)} questions to data/questions.parquet")

def export_agents():
    """Export all agents from FAISS to agents.parquet"""
    from agentsearch.utils.vector_store import load_index
    import os

    print("Exporting agents...")

    collection_name = 'agents'
    embeddings_path = os.path.join("faiss", f"{collection_name}_embeddings.npy")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

    _, metadata = load_index(collection_name)
    embeddings_array = np.load(embeddings_path)

    agents_data = []
    for idx, agent_id_str in enumerate(metadata['ids']):
        agent_id = int(agent_id_str)
        embedding = embeddings_array[idx].tolist()

        agent_record = {
            'id': agent_id,
            'embedding': embedding
        }
        agents_data.append(agent_record)

    agents_export_df = pd.DataFrame(agents_data)
    agents_export_df.to_parquet('data/agents.parquet', index=False)
    print(f"Exported {len(agents_export_df)} agents to data/agents.parquet")

if __name__ == "__main__":
    export_questions()
    export_agents()
    print("Export completed successfully!")