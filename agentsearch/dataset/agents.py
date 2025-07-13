from dataclasses import dataclass
from agentsearch.arxiv.type import ArxivPaper, ArxivCategory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from chromadb.api.types import QueryResult
import pandas as pd
from ast import literal_eval
import os
from agentsearch.agent.rag import retrieve
from agentsearch.agent import qa
from agentsearch.dataset.questions import questions_store
import numpy as np

db_location = "./chroma_db"
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

agents_df = pd.read_csv('data/authors.csv', index_col=0)
agents_df = agents_df[agents_df.index.astype(str).isin(os.listdir("papers/html"))]
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
    scholar_url: str
    papers: list[ArxivPaper]
    embedding: np.ndarray

    @classmethod
    def from_id(cls, id: int) -> 'Agent':
        agent = cls(
            id=id,
            name=agents_df.loc[id, 'name'],
            research_fields=agents_df.loc[id, 'research_fields'],
            scholar_url=agents_df.loc[id, 'scholar_url'],
            papers=[],
            embedding=None
        )
        agent.load_papers()
        agent.load_embedding()
        return agent
    
    @classmethod
    def all(cls) -> list['Agent']:
        agent_ids = agents_df.index.tolist()
        agents = [cls.from_id(id) for id in agent_ids]
        return agents
    
    def load_embedding(self):
        result = agents_store._collection.get(
            ids=[str(self.id)],
            include=['embeddings']
        )
        if len(result['embeddings']) == 0:
            raise ValueError(f"No embedding found for agent ID {self.id}")
        self.embedding = result['embeddings'][0]
    
    def load_papers(self) -> list[ArxivPaper]:
        self.papers = []
        papers_dir = f'papers/html/{self.name}'
        if not os.path.exists(papers_dir):
            return
        paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.html')]
        
        for paper_file in paper_files:
            paper_id = paper_file[len('arXiv-'):-len('.html')]
            try:
                paper = ArxivPaper.from_id(paper_id)
                self.papers.append(paper)
            except ValueError:
                continue

    def ask(self, question: str) -> str:
        sources = retrieve(self.id, question)
        sources_text = "\n".join([f"- {source.page_content}" for source in sources])
        if len(sources) == 0:
            sources_text = "No sources found"
        print("Number of sources:", len(sources))
        answer = qa.chain.invoke({"sources": sources_text, "question": question})
        return answer

@dataclass
class AgentMatch:
    agent: Agent
    similarity_score: float

def match_by_qid(qid: int, top_k: int = 1, blacklist: list[int] = []) -> list[AgentMatch]:
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
        ids=[str(id) for id in agents_df.index.tolist() if id not in blacklist]
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
