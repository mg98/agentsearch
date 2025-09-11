from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    with open("data/test_qids.txt", "r") as f:
        test_qids = [int(qid.strip()) for qid in f.read().split(',')]

    agent_store = AgentStore(use_llm_agent_card=True)

    scores = [] 
    for qid in test_qids:
        question = Question.from_id(qid)
        matches = agent_store.match_by_qid(qid, top_k=1)
        top_agent: Agent = matches[0].agent
        score = top_agent.has_sources(question.question)
        scores.append(score)

    print(f"Score: {np.sum(scores)}/{len(scores)}")