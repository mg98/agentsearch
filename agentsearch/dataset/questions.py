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
    agent_id: int | None
    text: str
    embedding: np.ndarray

    @classmethod
    def from_id(cls, id: int, shallow: bool = False) -> 'Question':
        agent_id_val = questions_df.loc[id, 'agent_id']
        question = cls(
            id=id,
            agent_id=int(agent_id_val) if pd.notna(agent_id_val) else None,
            text=questions_df.loc[id, 'question'],
            embedding=None
        )
        if not shallow:
            question.load_embedding()
        return question

    @classmethod
    def many(cls, ids: list[int], shallow: bool = False) -> list['Question']:
        questions = []
        for id in ids:
            agent_id_val = questions_df.loc[id, 'agent_id']
            question = cls(
                id=id,
                agent_id=int(agent_id_val) if pd.notna(agent_id_val) else None,
                text=questions_df.loc[id, 'question'],
                embedding=None
            )
            questions.append(question)

        if not shallow:
            # Load all embeddings in a single batch query
            question_ids = [str(q.id) for q in questions]
            if question_ids:
                result = questions_store._collection.get(
                    ids=question_ids,
                    include=['embeddings']
                )

                # Create a mapping from ID to embedding
                id_to_embedding = {}
                for idx, q_id in enumerate(result['ids']):
                    if idx < len(result['embeddings']):
                        id_to_embedding[q_id] = result['embeddings'][idx]

                # Assign embeddings to questions
                for question in questions:
                    str_id = str(question.id)
                    if str_id in id_to_embedding:
                        question.embedding = id_to_embedding[str_id]
                    else:
                        raise ValueError(f"No embedding found for question ID {question.id}")

        return questions
    
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
