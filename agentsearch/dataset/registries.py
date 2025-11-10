from agentsearch.dataset.agents import agents_df
import pandas as pd
from dataclasses import dataclass
from agentsearch.dataset.questions import Question
import numpy as np

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
    def match_by_qid(self, qid: int, top_k: int = 1) -> list[RegistryMatch]:
        from agentsearch.utils.vector_store import load_index

        question = Question.from_id(qid)
        question_embedding = np.array([question.embedding]).astype('float32')

        index, metadata = load_index('registries')
        distances, indices = index.search(question_embedding, len(metadata['ids']))

        root_matches: list[tuple[int, float]] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            if metadata['metadatas'][idx] and metadata['metadatas'][idx].get('parent') == 'root':
                root_matches.append((idx, distances[0][i]))

        root_matches.sort(key=lambda x: x[1])
        root_matches = root_matches[:top_k]

        matches: list[RegistryMatch] = []
        for idx, distance in root_matches:
            registry = metadata['documents'][idx]
            similarity_score = 1 - (distance / 2)
            matches.append(RegistryMatch(
                registry=registry,
                similarity_score=similarity_score
            ))

        return matches
