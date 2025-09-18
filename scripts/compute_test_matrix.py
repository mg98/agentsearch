from agentsearch.utils.eval import compute_question_agent_matrix, load_test_questions
from agentsearch.dataset.agents import AgentStore
import numpy as np

if __name__ == "__main__":
    print("Loading questions...")
    questions = load_test_questions()
    print("Loading agents...")
    agents = AgentStore(use_llm_agent_card=False).all(shallow=True)

    print("Computing matrix...")
    matrix = compute_question_agent_matrix(questions, agents)
    
    print("Saving matrix to disk...")
    np.save('data/test_matrix_raw.npy', matrix)
    print(f"Matrix saved with shape: {matrix.shape}")
    