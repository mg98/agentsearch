from dataclasses import dataclass
from langchain_chroma import Chroma
from chromadb.api.types import QueryResult
import pandas as pd
from ast import literal_eval
import os
from agentsearch.agent.rag import retrieve
from agentsearch.agent import qa
from agentsearch.dataset.questions import questions_store
import numpy as np
from agentsearch.dataset.papers import Paper
from agentsearch.utils.globals import db_location, embeddings

agents_df = pd.read_csv('data/authors.csv', index_col=0)
agents_df = agents_df[agents_df.index.astype(str).isin(os.listdir("papers/pdf"))]
agents_df['research_fields'] = agents_df['research_fields'].apply(literal_eval)
agents_df = agents_df[agents_df['research_fields'].apply(len) > 0]


agents_store = Chroma(
    collection_name="agents",
    persist_directory=db_location,
    embedding_function=embeddings
)

@dataclass
class Agent:
    id: int
    name: str
    research_fields: list[str]
    citation_count: int
    scholar_url: str
    papers: list[Paper]
    embedding: np.ndarray

    @classmethod
    def from_id(cls, id: int, shallow: bool = False) -> 'Agent':
        agent = cls(
            id=id,
            name=agents_df.loc[id, 'name'],
            research_fields=agents_df.loc[id, 'research_fields'],
            citation_count=agents_df.loc[id, 'citation_count'],
            scholar_url=agents_df.loc[id, 'scholar_url'],
            papers=[],
            embedding=None
        )
        if not shallow:
            agent.load_papers()
            agent.load_embedding()
        return agent
    
    @classmethod
    def all(cls, shallow: bool = False) -> list['Agent']:
        agent_ids = agents_df.index.tolist()
        agents = [cls.from_id(id, shallow) for id in agent_ids]
        return agents
    
    @classmethod
    def all_from_cluster(cls, topic: str, size: int) -> list['Agent']:
        # Get embedding for the topic
        topic_embedding = embeddings.embed_query(topic)
        
        # Query the agents_store for closest agents using the collection directly
        search_results: QueryResult = agents_store._collection.query(
            query_embeddings=[topic_embedding],
            n_results=size,
            include=['documents', 'distances']
        )
        
        # Convert results to Agent objects
        agents = []
        if search_results['ids'] is not None:
            for agent_id in search_results['ids'][0]:
                agent = cls.from_id(int(agent_id))
                agents.append(agent)
        
        return agents
    
    def load_embedding(self):
        result = agents_store._collection.get(
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


    def ask(self, question: str) -> int:
        sources = retrieve(self.id, question)
        return len(sources)
        sources_text = "\n".join([f"- {source.page_content}" for source in sources])
        if len(sources) == 0:
            sources_text = "No sources found"
            return "I don't know" # hotfix to make evaluation more efficient
        answer = qa.chain.invoke({"sources": sources_text, "question": question})
        return answer

@dataclass
class AgentMatch:
    agent: Agent
    similarity_score: float

def match_by_qid(qid: int, top_k: int = 1, whitelist: list[int] = []) -> list[AgentMatch]:
    """
    Match a question to the most similar agents based on Agent Card
    
    Args:
        question_id: The ID of the question to match
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

    question_embedding = question_result['embeddings'][0]
    search_results: QueryResult = agents_store._collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=['documents', 'distances'],
        ids=[str(id) for id in whitelist]
    )
    matches: list[AgentMatch] = []
    if search_results['distances'] is not None:
        for i, distance in enumerate(search_results['distances'][0]):
            agent_id = search_results['ids'][0][i]
            agent = Agent.from_id(int(agent_id))
            similarity_score = 1 - distance if distance <= 1 else 1 / (1 + distance)
            matches.append(AgentMatch(
                agent=agent,
                similarity_score=similarity_score
            ))
    
    return matches
