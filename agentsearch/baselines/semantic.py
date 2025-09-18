from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
import numpy as np
from tqdm import tqdm

def semantic_match(agent_store: AgentStore, question: Question) -> list[Agent]:
    matches = agent_store.match_by_qid(question.id, top_k=8)
    return list(map(lambda m: m.agent, matches))