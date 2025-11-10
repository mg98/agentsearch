from rank_bm25 import BM25Okapi
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
import numpy as np

def init_bm25(collection: str = "agents") -> BM25Okapi:
    agents = Agent.all(collection=collection)
    return BM25Okapi([agent.agent_card.replace(",", "").split(" ") for agent in agents])

def bm25_match(bm25: BM25Okapi, question: Question, collection: str = "agents") -> list[Agent]:
    agents = Agent.all(collection=collection)
    bm25_scores = bm25.get_scores(question.text.split(" "))
    sorted_indices = np.argsort(bm25_scores)[::-1]
    top_agents = [agents[i] for i in sorted_indices[:8]]
    return top_agents
