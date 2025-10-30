from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question

def semantic_match(agent_store: AgentStore, question: Question, top_k: int = 8) -> list[Agent]:
    matches = agent_store.match_by_qid(question.id, top_k=top_k)
    return list(map(lambda m: m.agent, matches))