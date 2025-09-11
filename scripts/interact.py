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
K_MATCHES = 16

def simulate_interactions(
        agent_store: AgentStore, 
        core_agent_questions: list[Question], 
        observed_agents_with_questions: list[tuple[Agent, list[Question]]],
        attack_vol: float
        ) -> list[Interaction]:
    edges: list[Interaction] = []

    for question in tqdm(core_agent_questions, desc="Core agent asking questions"):
        print(f"\n{Fore.GREEN}Core agent asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
        
        matches = agent_store.match_by_qid(question.id, K_MATCHES)
        assert len(matches) == K_MATCHES
        for match in matches:
            trust_score = match.agent.get_confidence(question.question)
            print(f"Core agent asked {match.agent.name}: {trust_score}")
            edges.append(Interaction(Agent.make_dummy(), match.agent, question, trust_score))
            if trust_score > 0:
                break

    num_adversarial = int(len(observed_agents_with_questions) * attack_vol)

    for agent, questions in tqdm(observed_agents_with_questions[:num_adversarial], desc="Observed adversarial agents asking questions"):
        for question in tqdm(questions, desc="Agent asking questions"):
            print(f"\n{Fore.GREEN}Agent {agent.name} asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
                
            matches = agent_store.match_by_qid(question.id, K_MATCHES)
            for match in matches:
                trust_score = 0 if random.random() < 0.5 else random.random()
                print(f"{agent.name} asked {match.agent.name}: {trust_score}")
                edges.append(Interaction(agent, match.agent, question, trust_score))
                if trust_score > 0:
                    break

    for agent, questions in tqdm(observed_agents_with_questions[num_adversarial:], desc="Observed benign agents asking questions"):
        for question in tqdm(questions, desc="Agent asking questions"):
            print(f"\n{Fore.GREEN}Agent {agent.name} asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
                
            matches = agent_store.match_by_qid(question.id, K_MATCHES)
            for match in matches:
                trust_score = match.agent.get_confidence(question.question)
                print(f"{agent.name} asked {match.agent.name}: {trust_score}")
                edges.append(Interaction(agent, match.agent, question, trust_score))
                if trust_score > 0:
                    break

    return edges

if __name__ == '__main__':
    agent_store = AgentStore(use_llm_agent_card=False)
    all_agents = agent_store.all(shallow=True)
    random.shuffle(all_agents)
    observed_agents = all_agents[:int(len(all_agents) / np.log(len(all_agents)))] # N / log(N)

    all_questions = Question.all()
    random.shuffle(all_questions)

    test_questions = all_questions[:1000]
    core_agent_questions = all_questions[1000:1100]
    observed_agent_questions = all_questions[1100:]

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


    for attack_vol in np.arange(0.0, 1.1, 0.1):
        edges = simulate_interactions(agent_store, core_agent_questions, observed_agents_with_questions, attack_vol)
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
        edges_df.to_csv(f'data/graph/edges_{int(attack_vol*100)}.csv', index=False)

    print(f"\n{Fore.GREEN}Saved graph data.{Style.RESET_ALL}")
