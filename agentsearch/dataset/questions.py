import pandas as pd
import numpy as np
from agentsearch.utils.vector_store import get_question_embedding
from agentsearch.utils.globals import embeddings
import os
import warnings

if not os.path.exists('data/questions.csv'):
    warnings.warn("data/questions.csv not found, using empty DataFrame for questions_df")
    questions_df = pd.DataFrame(columns=['agent_id', 'question'])
else:
    questions_df = pd.read_csv('data/questions.csv', index_col=0)
    questions_df = questions_df[questions_df['question'].notna()].copy()

class Question:
    def __init__(self, id: int, agent_id: int | None, text: str):
        self.id = id
        self.agent_id = agent_id
        self.text = text
        self._embedding = None

    @property
    def embedding(self) -> np.ndarray:
        if self._embedding is None:
            self._embedding = get_question_embedding(self.id)
        return self._embedding

    @classmethod
    def from_id(cls, id: int) -> 'Question':
        agent_id_val = questions_df.loc[id, 'agent_id']
        return cls(
            id=id,
            agent_id=int(agent_id_val) if pd.notna(agent_id_val) else None,
            text=questions_df.loc[id, 'question']
        )

    @classmethod
    def many(cls, ids: list[int]) -> list['Question']:
        questions = []
        for id in ids:
            agent_id_val = questions_df.loc[id, 'agent_id']
            question = cls(
                id=id,
                agent_id=int(agent_id_val) if pd.notna(agent_id_val) else None,
                text=questions_df.loc[id, 'question']
            )
            questions.append(question)
        return questions

    @classmethod
    def all(cls, from_agents: list[int] | None = None) -> list['Question']:
        if from_agents is None:
            return [cls.from_id(idx) for idx in questions_df.index]

        questions = []
        for agent_id in from_agents:
            agent_questions = questions_df[questions_df['agent_id'] == agent_id]
            for idx in agent_questions.index:
                questions.append(cls.from_id(idx))
        return questions

    @classmethod
    def all_from_cluster(cls, topic: str, size: int) -> list['Question']:
        topic_embedding = np.array(embeddings.embed_query(topic))

        all_questions = cls.all()

        similarities = []
        for question in all_questions:
            cos_sim = np.dot(question.embedding, topic_embedding) / (
                np.linalg.norm(question.embedding) * np.linalg.norm(topic_embedding)
            )
            similarities.append((question, cos_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [q for q, _ in similarities[:size]]
