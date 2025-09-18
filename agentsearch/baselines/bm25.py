from rank_bm25 import BM25Okapi
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
import numpy as np

def bm25_match(agent_store: AgentStore, question: Question) -> list[Agent]:
    agents = agent_store.all(shallow=True)
    bm25 = BM25Okapi([agent.agent_card.replace(",", "").split(" ") for agent in agents])
    bm25_scores = bm25.get_scores(question.question.split(" "))
    sorted_indices = np.argsort(bm25_scores)[::-1]
    top_agents = [agents[i] for i in sorted_indices[:8]]
    return top_agents
