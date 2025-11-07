from agentsearch.utils.eval import compute_question_agent_matrix, load_test_questions
from agentsearch.dataset.agents import Agent
import numpy as np

if __name__ == "__main__":
    print("Loading questions...")
    questions = load_test_questions()
    print("Loading agents...")
    agents = Agent.all()

    print("Computing matrix...")
    matrix = compute_question_agent_matrix(questions, agents)

    print("Saving raw matrix to disk...")
    np.save('data/test_matrix_raw.npy', matrix)
    print(f"Raw matrix saved with shape: {matrix.shape}")

    print("Creating binary matrix (scores > 0 → 1)...")
    binary_matrix = (matrix > 0).astype(np.int8)
    np.save('data/test_matrix.npy', binary_matrix)
    print(f"Binary matrix saved with shape: {binary_matrix.shape}")
    