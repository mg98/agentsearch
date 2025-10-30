from dataclasses import dataclass
from langchain_chroma import Chroma
from chromadb.api.types import QueryResult
import pandas as pd
from ast import literal_eval
import os
from agentsearch.agent.rag_faiss import retrieve, retrieve_with_embedding
from agentsearch.agent import qa
from agentsearch.dataset.questions import questions_store
import numpy as np
from agentsearch.dataset.papers import Paper
from agentsearch.utils.globals import db_location, embeddings
import warnings
from agentsearch.dataset.questions import Question
import faiss
import json

agents_df = pd.read_csv('data/agents.csv', index_col=0)
if os.path.exists("papers/pdf"):
    agents_df = agents_df[agents_df.index.astype(str).isin(os.listdir("papers/pdf"))]
else:
    warnings.warn("no papers/pdf directory found")
agents_df['research_fields'] = agents_df['research_fields'].apply(literal_eval)
agents_df = agents_df[agents_df['research_fields'].apply(len) > 0]
agents_df = agents_df.sample(frac=1, random_state=42)

# Load LLM-generated agent cards
agentcards_df = pd.read_csv('data/agentcards.csv', index_col=0)
agentcards_df = agentcards_df.reindex(agents_df.index)

def num_sources_to_score(num_sources: int) -> float:
    if num_sources >= 100:
        return 1.0
    return np.log(num_sources + 1) / np.log(101)

@dataclass
class Agent:
    id: int
    name: str
    citation_count: int
    scholar_url: str
    agent_card: str
    faculty: str
    department: str
    group: str
    papers: list[Paper]
    embedding: np.ndarray

    _store: Chroma | None = None

    @classmethod
    def make_dummy(cls, id=0) -> 'Agent':
        return Agent(
            id=id,
            name="Dummy Agent",
            citation_count=0,
            scholar_url="",
            agent_card="",
            faculty="",
            department="",
            group="",
            papers=[],
            embedding=None,
            _store=None
        )
    
    def load_embedding(self):
        result = self._store._collection.get(
            ids=[str(self.id)],
            include=['embeddings']
        )
        if len(result['embeddings']) == 0:
            raise ValueError(f"No embedding found for agent ID {self.id}")
        self.embedding = result['embeddings'][0]
    
    def load_papers(self) -> list[Paper]:
        papers_dir = f'papers/pdf/{self.id}'
        if not os.path.exists(papers_dir):
            self.papers = []
        paper_ids = [f[:-len('.pdf')] for f in os.listdir(papers_dir) if f.endswith('.pdf')]
        self.papers = [Paper(id=paper_id, agent_id=self.id) for paper_id in paper_ids]


    def ask(self, question: str) -> str:
        sources = retrieve(self.id, question)
        sources_text = "\n".join([f"- {source.page_content}" for source in sources])
        if len(sources) == 0:
            sources_text = "No sources found"
            return "I don't know" # hotfix to make evaluation more efficient
        answer = qa.chain.invoke({"sources": sources_text, "question": question})
        return answer
    
    def count_sources(self, question: str) -> int:
        sources = retrieve(self.id, question)
        return len(sources)
    
    def has_sources(self, question: str) -> bool:
        sources = retrieve(self.id, question, k=1)
        return len(sources) > 0

    def grade(self, question: Question) -> float:
        sources = retrieve_with_embedding(self.id, question.embedding, k=100)
        return num_sources_to_score(len(sources))

@dataclass
class AgentMatch:
    agent: Agent
    distance: float

class AgentStore:
    def __init__(self, use_llm_agent_card: bool):
        self.use_llm_agent_card = use_llm_agent_card
        collection_name = "agents_with_llm_agent_cards" if use_llm_agent_card else "agents_with_human_agent_cards"
        self._store = Chroma(
            collection_name=collection_name,
            persist_directory=db_location,
            embedding_function=embeddings
        )
    

    def from_id(self, id: int, shallow: bool = False) -> 'Agent':
        if self.use_llm_agent_card:
            agent_card = agentcards_df.loc[id, 'agent_card']
            if type(agent_card) != str:
                agent_card = ', '.join(agents_df.loc[id, 'research_fields'])
        else:
            agent_card = ', '.join(agents_df.loc[id, 'research_fields'])

        agent = Agent(
            id=id,
            name=agents_df.loc[id, 'name'],
            citation_count=agents_df.loc[id, 'citation_count'],
            scholar_url=agents_df.loc[id, 'scholar_url'],
            agent_card=agent_card,
            faculty=agents_df.loc[id, 'faculty'],
            department=agents_df.loc[id, 'department'],
            group=agents_df.loc[id, 'group'],
            papers=[],
            embedding=None,
            _store=self._store
        )

        if not shallow:
            agent.load_papers()
            agent.load_embedding()
        
        return agent
    
    def all(self, shallow: bool = False) -> list[Agent]:
        agent_ids = agents_df.index.tolist()
        agents = [self.from_id(id, shallow) for id in agent_ids]
        return agents
    
    def all_from_cluster(self, topic: str, size: int) -> list[Agent]:
        topic_embedding = embeddings.embed_query(topic)
        
        # Query the agents_store for closest agents using the collection directly
        search_results: QueryResult = self._store._collection.query(
            query_embeddings=[topic_embedding],
            n_results=size,
            include=['documents', 'distances']
        )
        
        # Convert results to Agent objects
        agents = []
        if search_results['ids'] is not None:
            for agent_id in search_results['ids'][0]:
                agent = self.from_id(int(agent_id))
                agents.append(agent)
        
        return agents

    def match_by_qid(self, qid: int, top_k: int = 1) -> list[AgentMatch]:
        """
        Match a question to the most similar agents using FAISS

        Args:
            qid: The ID of the question to match
            top_k: Number of top matches to return

        Returns:
            List of agent matches
        """
        question_result = questions_store._collection.get(
            ids=[str(qid)],
            include=['embeddings']
        )
        if len(question_result['embeddings']) == 0:
            raise ValueError(f"No embedding found for question ID {qid}")

        question_embedding = np.array([question_result['embeddings'][0]]).astype('float32')

        collection_name = "agents_with_llm_agent_cards" if self.use_llm_agent_card else "agents_with_human_agent_cards"
        index_path = os.path.join("faiss", f"{collection_name}.bin")
        meta_path = os.path.join("faiss", f"{collection_name}_meta.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        index = faiss.read_index(index_path)

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        distances, indices = index.search(question_embedding, top_k)

        matches: list[AgentMatch] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            agent_id = int(metadata['ids'][idx])
            distance = float(distances[0][i])
            agent = self.from_id(agent_id)
            matches.append(AgentMatch(
                agent=agent,
                distance=distance
            ))

        return matches