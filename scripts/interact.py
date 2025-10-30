import random
import numpy as np
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent, AgentStore
from agentsearch.dataset.questions import Question
from agentsearch.graph.utils import  Interaction
from tqdm import tqdm
import pandas as pd

init(autoreset=True)
random.seed(42)
K_MATCHES = 8

def simulate_interactions(
        agent_store: AgentStore, 
        questions: list[Question]
        ) -> list[Interaction]:
    edges: list[Interaction] = []

    for question in tqdm(questions, desc="Core agent asking questions"):
        print(f"\n{Fore.GREEN}Asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
        
        matches = agent_store.match_by_qid(question.id, K_MATCHES)
        assert len(matches) == K_MATCHES
        for i, match in enumerate(matches):
            trust_score = match.agent.grade(question)
            print(f"({i+1}) Asked {match.agent.name}: {trust_score}")
            edges.append(Interaction(match.agent, question, trust_score))
            if trust_score > 0:
                break

    return edges

if __name__ == '__main__':
    agent_store = AgentStore(use_llm_agent_card=False)
    all_agents = agent_store.all(shallow=True)
    random.shuffle(all_agents)

    all_questions = Question.all()
    random.shuffle(all_questions)

    test_questions = all_questions[:1000]
    rest_questions = all_questions[1100:]

    # Write test_core_agent_questions IDs to file for evaluation
    test_qids = [str(q.id) for q in test_questions]
    with open('data/test_qids.txt', 'w') as f:
        f.write(','.join(test_qids))
    print(f"{Fore.CYAN}Wrote {len(test_qids)} test question IDs to data/test_qids.txt{Style.RESET_ALL}")


    edges = simulate_interactions(agent_store, rest_questions)
    random.shuffle(edges)

    edges_df = pd.DataFrame([
        {
            'agent': edge.target_agent.id,
            'question': edge.question.id,
            'score': edge.score
        }
        for edge in edges
    ])
    edges_df.to_csv(f'data/graph/reports.csv', index=False)

    print(f"\n{Fore.GREEN}Saved graph data.{Style.RESET_ALL}")
