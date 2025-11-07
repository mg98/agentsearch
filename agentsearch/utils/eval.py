import pandas as pd
from agentsearch.dataset.questions import Question
from agentsearch.dataset.agents import Agent, agents_df
import numpy as np
from sklearn.metrics import ndcg_score
from dataclasses import dataclass
from tqdm import tqdm
from agentsearch.utils.globals import THRESHOLD
import os


def load_data(edges_path: str) -> pd.DataFrame:
    return pd.read_csv(edges_path, dtype={
        'agent': int, 
        'question': int, 
        'score': float
    })

def load_test_questions() -> list[Question]:
    with open('data/test_qids.txt', 'r') as f:
        test_qids = [int(qid.strip()) for qid in f.read().split(',')]
    return Question.many(test_qids)

class TestOracle:
    """
    Oracle for the test questions.
    Retrieves agents based on their real and verified abilities to answer a test question.
    """
    def __init__(self):
        self.matrix = np.load('data/test_matrix.npy')
        with open('data/test_qids.txt', 'r') as f:
            self.test_qids = [int(qid.strip()) for qid in f.read().split(',')]

    def _map_question_id(self, question_id: int) -> int:
        """
        Returns row index matching question ID.
        """
        return self.test_qids.index(question_id)

    def _map_agent_id(self, agent_id: int) -> int:
        """
        Returns column index matching agent ID.
        """
        return list(agents_df.index).index(agent_id)

    def get_score(self, question_id: int, agent_id: int) -> float:
        """
        Returns the score for a given question ID and agent ID.
        """
        return self.matrix[
            self._map_question_id(question_id), 
            self._map_agent_id(agent_id)
            ].item()

    def rank_agent_ids(self, question_id: int, top_k=8) -> list[int]:
        """
        Returns top-k agent IDs for a given question ID.
        """
        question_scores = self.matrix[self._map_question_id(question_id)]
        top_agent_indices = np.argsort(question_scores)[::-1][:top_k]
        return agents_df.iloc[top_agent_indices].index.tolist()

    def rank_agents(self, question: Question, top_k=8, collection: str = "agents") -> list[Agent]:
        """
        Wrapper around `rank_agent_ids` that returns `Agent` objects.
        """
        agent_ids = self.rank_agent_ids(question.id, top_k)
        return list(map(lambda id: Agent.from_id(id, collection=collection), agent_ids))

@dataclass
class EvalMetrics:
    precision_1: float
    precision_3: float
    rr_8: float
    ndcg_8: float

@dataclass
class MatchResult:
    rank: int # given by match_fn
    agent: Agent
    score: float # real score

def compute_metrics(match_results: list[MatchResult]) -> EvalMetrics:
    score_1 = match_results[0].score
    precision_1 = int(match_results[0].score > 0)
    precision_3 = sum([int(score > 0) for score in match_results[:3]]) / 3
    rr_8 = next((1 / (match_result.rank + 1) for match_result in match_results if match_result.score > 0), 0)
    
    y_pred = [8 - match_result.rank for match_result in match_results]
    y_true = [match_result.score for match_result in match_results]
    ndcg_8 = ndcg_score(y_true, y_pred, k=8)

    return EvalMetrics(precision_1, precision_3, rr_8, ndcg_8)


def compute_question_agent_matrix(questions: list[Question], agents: list[Agent]) -> np.ndarray:
    from agentsearch.utils.vector_store import load_index

    matrix = np.zeros((len(questions), len(agents)))

    for agent_idx, agent in enumerate(tqdm(agents, desc="Processing agents")):
        collection_name = f"agent_{agent.id}"

        try:
            index, _ = load_index(collection_name)
        except FileNotFoundError:
            continue

        query_embeddings = np.array([q.embedding for q in questions]).astype('float32')
        distances, _ = index.search(query_embeddings, 100)

        for question_idx in range(len(questions)):
            num_sources = sum(1 for distance in distances[question_idx] if distance < THRESHOLD)
            matrix[question_idx, agent_idx] = num_sources

    return matrix
