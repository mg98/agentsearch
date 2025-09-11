import pandas as pd
from langchain_chroma import Chroma
from dataclasses import dataclass
import numpy as np
from agentsearch.utils.globals import db_location, embeddings
from chromadb.api.types import QueryResult

questions_df = pd.read_csv('data/questions.csv', index_col=0)
questions_store = Chroma(
            collection_name='questions',
            persist_directory=db_location,
            embedding_function=embeddings
        )

@dataclass
class Question:
    id: int
    agent_id: int
    question: str
    embedding: np.ndarray

    @classmethod
    def from_id(cls, id: int, shallow: bool = False) -> 'Question':
        question = cls(
            id=id,
            agent_id=int(questions_df.loc[id, 'agent_id']),
            question=questions_df.loc[id, 'question'],
            embedding=None
        )
        if not shallow:
            question.load_embedding()
        return question
    
    @classmethod
    def all(cls, from_agents: list[int] | None = None, questions_per_agent: int = 100, shallow: bool = False) -> list['Question']:
        if from_agents is None:
            return [cls.from_id(idx, shallow) for idx in questions_df.index]
        
        questions = []
        for agent_id in from_agents:
            agent_questions = questions_df[questions_df['agent_id'] == agent_id]
            for idx in agent_questions.index[:questions_per_agent]:
                question = cls.from_id(idx, shallow)
                questions.append(question)
        return questions
    
    @classmethod
    def all_from_cluster(cls, topic: str, size: int) -> list['Question']:
        # Get embedding for the topic
        topic_embedding = embeddings.embed_query(topic)
        
        # Query the agents_store for closest agents using the collection directly
        search_results: QueryResult = questions_store._collection.query(
            query_embeddings=[topic_embedding],
            n_results=size,
            include=['documents', 'distances']
        )
        
        # Convert results to Agent objects
        questions = []
        if search_results['ids'] is not None:
            for question_id in search_results['ids'][0]:
                question = cls.from_id(int(question_id))
                questions.append(question)
        
        return questions

    def load_embedding(self):
        result = questions_store._collection.get(
            ids=[str(self.id)],
            include=['embeddings']
        )
        if len(result['embeddings']) == 0:
            raise ValueError(f"No embedding found for question ID {self.id}")
        self.embedding = result['embeddings'][0]
