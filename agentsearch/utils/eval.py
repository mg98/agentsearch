import pandas as pd
from typing import Callable
from agentsearch.dataset.questions import Question
from agentsearch.dataset.agents import Agent, AgentStore, agents_df, num_sources_to_score
import numpy as np
from sklearn.metrics import ndcg_score
from dataclasses import dataclass
from tqdm import tqdm
from langchain_chroma import Chroma
from agentsearch.utils.globals import db_location, embeddings


def load_data(edges_path: str) -> pd.DataFrame:
    return pd.read_csv(edges_path, dtype={
        'source_agent': int, 
        'target_agent': int, 
        'question': int, 
        'score': float
    })

def load_all_attacks() -> list[tuple[int, pd.DataFrame]]:
    data = []
    for i in range(0, 101, 10):
        data.append((i, load_data(f'data/graph/edges_{i}.csv')))
    return data

def load_test_questions() -> list[Question]:
    with open('data/test_qids.txt', 'r') as f:
        test_qids = [int(qid.strip()) for qid in f.read().split(',')]
    return Question.many(test_qids, shallow=False)

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

    def rank_agents(self, agent_store: AgentStore, question: Question, top_k=8) -> list[Agent]:
        """
        Wrapper around `rank_agent_ids` that returns `Agent` objects.
        """
        agent_ids = self.rank_agent_ids(question.id, top_k)
        return list(map(lambda id: agent_store.from_id(id, shallow=True), agent_ids))

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
    matrix = np.zeros((len(questions), len(agents)))
    for agent_idx, agent in enumerate(tqdm(agents, desc="Processing agents")):
        vector_store = Chroma(
            collection_name=f"agent_{agent.id}",
            persist_directory=db_location,
            embedding_function=embeddings
        )
        results = vector_store._collection.query(
            query_embeddings=np.array([q.embedding for q in questions]),
            n_results=100,
            include=['distances']
        )

        # Fill matrix with number of results for each question-agent pair
        for question_idx in range(len(questions)):
            # Filter results by similarity threshold (1 - distance >= 0.5)
            distances = results['distances'][question_idx]
            num_sources = sum(1 for distance in distances if (1 - distance) >= 0.5)
            matrix[question_idx, agent_idx] = num_sources #num_sources_to_score(num_sources)
            
    return matrix
