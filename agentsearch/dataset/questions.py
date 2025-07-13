import pandas as pd
from ast import literal_eval
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dataclasses import dataclass
import numpy as np

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

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
    def from_id(cls, id: int) -> 'Question':
        question = cls(
            id=id,
            agent_id=int(questions_df.loc[id, 'agent_id']),
            question=questions_df.loc[id, 'question'],
            embedding=None
        )
        question.load_embedding()
        return question
    
    @classmethod
    def all(cls) -> list['Question']:
        questions = []
        for idx, _ in questions_df.iterrows():
            question = cls.from_id(idx)
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
