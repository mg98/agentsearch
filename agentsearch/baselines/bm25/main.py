from rank_bm25 import BM25Okapi
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    with open("data/test_qids.txt", "r") as f:
        test_qids = [int(qid.strip()) for qid in f.read().split(',')]
    
    test_questions: list[Question] = []
    for qid in test_qids:
        question = Question.from_id(qid)
        test_questions.append(question)

    agent_store = AgentStore(use_llm_agent_card=True)
    agents = agent_store.all(shallow=True)
    bm25 = BM25Okapi([agent.agent_card.replace(",", "").split(" ") for agent in agents])
    
    scores = []
    for question in tqdm(test_questions, desc="Evaluating"):
        bm25_scores = bm25.get_scores(question.question.split(" "))
        top_agent: Agent = agents[np.argmax(bm25_scores)]
        score = top_agent.has_sources(question.question)
        scores.append(score)

    print(f"Score: {np.sum(scores)}/{len(scores)}")