from rank_bm25 import BM25Okapi
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
import numpy as np

def init_bm25(agent_store: AgentStore) -> BM25Okapi:
    agents = agent_store.all(shallow=True)
    return BM25Okapi([agent.agent_card.replace(",", "").split(" ") for agent in agents])

def bm25_match(bm25: BM25Okapi, agent_store: AgentStore, question: Question) -> list[Agent]:
    agents = agent_store.all(shallow=True)
    bm25_scores = bm25.get_scores(question.text.split(" "))
    sorted_indices = np.argsort(bm25_scores)[::-1]
    top_agents = [agents[i] for i in sorted_indices[:8]]
    return top_agents
