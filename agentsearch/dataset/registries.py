from langchain_chroma import Chroma
from agentsearch.dataset.agents import agents_df
from agentsearch.utils.globals import db_location, embeddings
import pandas as pd
from dataclasses import dataclass
from chromadb.api.types import QueryResult
from agentsearch.dataset.questions import questions_store

registries_store = Chroma(
    collection_name='registries',
    persist_directory=db_location,
    embedding_function=embeddings
)

def get_faculties():
    return agents_df['faculty'].unique().tolist()

def get_departments(faculty: str):
    return agents_df[agents_df['faculty'] == faculty]['department'].unique().tolist()

def get_groups(faculty: str, department: str):
    groups = agents_df[
        (agents_df['faculty'] == faculty) & (agents_df['department'] == department)
    ]['group'].unique().tolist()
    # Replace nan/NaN/None with 'unspecified'
    groups = ['unspecified' if isinstance(g, float) and pd.isna(g) else g for g in groups]
    return groups

@dataclass
class RegistryMatch:
    registry: str
    similarity_score: float

class RegistryStore:
    def __init__(self):
        self._store = Chroma(
            collection_name='registries',
            persist_directory=db_location,
            embedding_function=embeddings
        )
    
    def match_by_qid(self, qid: int, top_k: int = 1) -> list[RegistryMatch]:
        question_result = questions_store._collection.get(
            ids=[str(qid)],
            include=['embeddings']
        )
        if len(question_result['embeddings']) == 0:
            raise ValueError(f"No embedding found for question ID {qid}")

        question_embedding = question_result['embeddings'][0]
        search_results: QueryResult = self._store._collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=['documents', 'distances'],
            where={"parent": "root"}
        )
        matches: list[RegistryMatch] = []
        if search_results['distances'] is not None:
            for i, distance in enumerate(search_results['distances'][0]):
                registry = search_results['ids'][0][i]
                similarity_score = 1 - (distance / 2)
                matches.append(RegistryMatch(
                    registry=registry,
                    similarity_score=similarity_score
                ))
        
        return matches
