import pandas as pd
from agentsearch.dataset.questions import questions_df, Question
from agentsearch.dataset.agents import agents_df, Agent, AgentStore
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from agentsearch.baselines.bhopale.reranker import BERTCrossEncoderReranker, fine_tune_reranker

Data = list[str, str, float] # Question text, agent card, score

def evaluate(reranker: BERTCrossEncoderReranker, agent_store: AgentStore, test_questions: list[tuple[int, str]]):
    scores = []
    for qid, question_text in tqdm(test_questions, desc="Evaluating"):
        initial_matches = agent_store.match_by_qid(qid, top_k=100)
        reranked_matches = reranker.rerank_with_agents(
            query=question_text,
            agent_matches=initial_matches,
            top_k=1,
            show_progress=False
        )

        agent: Agent = reranked_matches[0].agent
        score = agent.has_sources(question_text)
        scores.append(score)

    print(f"Score: {np.sum(scores)}/{len(scores)}")

if __name__ == "__main__":
    with open("data/test_qids.txt", "r") as f:
        test_qids = [int(qid.strip()) for qid in f.read().split(',')]
    
    test_questions = []
    for qid in test_qids:
        question_text = questions_df.loc[qid, 'question']
        test_questions.append((qid, question_text))

    # Initialize agent store and reranker
    agent_store = AgentStore(use_llm_agent_card=False)
    
    for attack_volume in range(0, 101, 10):
        graph_data = pd.read_csv(f"data/graph_{attack_volume}.csv", 
                                 dtype={'source_agent': int, 'target_agent': int, 'question': int, 'score': float})
        data: list[Data] = []
        for _, row in graph_data.iterrows():
            question_text = questions_df.loc[row['question'], 'question']
            agent = agent_store.from_id(row['target_agent'], shallow=True)
            score = row['score']
            data.append((question_text, agent.agent_card, score))

        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        reranker = fine_tune_reranker(train_data, val_data)
        evaluate(reranker, agent_store, test_questions)