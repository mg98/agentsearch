import random
import pickle
import numpy as np
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent, AgentStore
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.graph.utils import visualize_graph, compute_trust_score, Interaction
from tqdm import tqdm
import pandas as pd

init(autoreset=True)
random.seed(42)
K_MATCHES = 8

def simulate_interactions(
        agent_store: AgentStore, 
        core_agent_questions: list[Question], 
        observed_agents_with_questions: list[tuple[Agent, list[Question]]]
        ) -> list[Interaction]:
    edges: list[Interaction] = []

    for question in tqdm(core_agent_questions, desc="Core agent asking questions"):
        print(f"\n{Fore.GREEN}Core agent asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
        
        matches = agent_store.match_by_qid(question.id, K_MATCHES, whitelist=[a.id for a in all_agents if a.id != core_agent.id])
        assert len(matches) == K_MATCHES
        for match in matches:
            trust_score = compute_trust_score(match.agent.count_sources(question.question))
            print(f"{core_agent.name} asked {match.agent.name}: {trust_score}")
            edges.append(Interaction(core_agent, match.agent, question, trust_score))

    for agent, questions in tqdm(observed_agents_with_questions, desc="Observed agents asking questions"):
        for question in tqdm(questions, desc="Agent asking questions"):
            print(f"\n{Fore.GREEN}Agent {agent.name} asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
                
            matches = agent_store.match_by_qid(question.id, K_MATCHES, whitelist=[a.id for a in all_agents if a.id != agent.id])
            assert len(matches) == K_MATCHES
            for match in matches:
                trust_score = compute_trust_score(match.agent.count_sources(question.question))
                print(f"{agent.name} asked {match.agent.name}: {trust_score}")
                edges.append(Interaction(agent, match.agent, question, trust_score))

    return edges

if __name__ == '__main__':
    agent_store = AgentStore(use_llm_agent_card=False)
    all_agents = agent_store.all()
    random.shuffle(all_agents)

    core_agent = all_agents[0]
    observed_agents = all_agents[1:2*int(len(all_agents) / np.log(len(all_agents)))] # 2N / log(N)

    all_questions = Question.all()
    random.shuffle(all_questions)

    test_questions = all_questions[:100]
    core_agent_questions = all_questions[100:200]
    observed_agent_questions = all_questions[200:]

    weights = np.random.random(len(observed_agents))
    weights = weights / weights.sum() # normalize to sum to 1
    questions_per_agent = np.round(weights * len(observed_agent_questions)).astype(int)
    
    # Map agent IDs to their allocated questions
    observed_agents_with_questions: list[tuple[Agent, list[Question]]] = []
    start_idx = 0
    for agent, num_questions in zip(observed_agents, questions_per_agent):
        end_idx = start_idx + num_questions
        observed_agents_with_questions.append((agent, observed_agent_questions[start_idx:end_idx]))
        start_idx = end_idx
    
    # Write test_core_agent_questions IDs to file for evaluation
    test_qids = [str(q.id) for q in test_questions]
    with open('data/test_qids.txt', 'w') as f:
        f.write(','.join(test_qids))
    print(f"{Fore.CYAN}Wrote {len(test_qids)} test question IDs to data/test_qids.txt{Style.RESET_ALL}")


    edges = simulate_interactions(agent_store, core_agent_questions, observed_agents_with_questions)
    random.shuffle(edges)

    edges_df = pd.DataFrame([
        {
            'source_agent': edge.source_agent.id,
            'target_agent': edge.target_agent.id,
            'question': edge.question.id,
            'score': edge.score
        }
        for edge in edges
    ])
    
    # for attack_vol in np.arange(0.0, 1.1, 0.1):
    #     edges_data = edges_df.copy()
    #     malicious_agent_ids = [agent.id for agent in observed_agents[:int(len(observed_agents) * attack_vol)]]
        
    #     # Flip scores for edges from malicious agents
    #     malicious_mask = edges_data['source_agent'].isin(malicious_agent_ids)
    #     edges_data.loc[malicious_mask, 'score'] = 1 - edges_data.loc[malicious_mask, 'score']
        
    #     # Save to CSV
    #     edges_data.to_csv(f'data/graph_{int(attack_vol*100)}.csv', index=False)

    print(f"\n{Fore.GREEN}Saved graph data.{Style.RESET_ALL}")
