import random
import numpy as np
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent, AgentStore
from agentsearch.dataset.questions import Question
from agentsearch.graph.utils import  Interaction
from tqdm import tqdm
import pandas as pd
import argparse

init(autoreset=True)
random.seed(42)
K_MATCHES = 16

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate agent-question interaction reports')
    parser.add_argument('--job-id', type=int, default=0, help='Job ID for parallel processing (0-indexed)')
    parser.add_argument('--job-count', type=int, default=1, help='Total number of parallel jobs')
    args = parser.parse_args()

    agent_store = AgentStore(use_llm_agent_card=False)
    all_agents = agent_store.all(shallow=True)
    random.shuffle(all_agents)
    
    all_questions = Question.all()
    random.shuffle(all_questions)

    test_questions = all_questions[:1000]
    questions = all_questions[1000:]

    # Write test_core_agent_questions IDs to file for evaluation
    test_qids = [str(q.id) for q in test_questions]
    with open('data/test_qids.txt', 'w') as f:
        f.write(','.join(test_qids))
    print(f"{Fore.CYAN}Wrote {len(test_qids)} test question IDs to data/test_qids.txt{Style.RESET_ALL}")


    edges = []

    for idx, question in enumerate(tqdm(questions, desc="Generating reports")):
        if idx % args.job_count != args.job_id:
            continue
        
        matches = agent_store.match_by_qid(question.id, K_MATCHES)
        for match in matches:
            trust_score = match.agent.grade(question)
            edges.append(Interaction(Agent.make_dummy(), match.agent, question, trust_score))
            if trust_score > 0:
                break

    random.shuffle(edges)

    edges_df = pd.DataFrame([
        {
            'agent': edge.target_agent.id,
            'question': edge.question.id,
            'score': edge.score
        }
        for edge in edges
    ])
    edges_df.to_csv(f'data/reports_{args.job_id}.csv', index=False)

    print(f"\n{Fore.GREEN}Saved graph data.{Style.RESET_ALL}")
