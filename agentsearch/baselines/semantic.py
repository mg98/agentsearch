from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question

def semantic_match(question: Question, top_k: int = 8, collection: str = "agents") -> list[Agent]:
    matches = Agent.match(question, top_k=top_k, collection=collection)
    return list(map(lambda m: m.agent, matches))