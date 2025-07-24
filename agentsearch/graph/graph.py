import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pickle
import os
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.agent import eval
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.graph.training import train_model, evaluate_and_predict, predict_top_targets, predict_trust_scores
from agentsearch.graph.utils import visualize_graph
from agentsearch.utils.globals import EMBEDDING_DIM
import sys
from tqdm import tqdm

# Initialize colorama
init(autoreset=True)

random.seed(42)


if __name__ == '__main__':
    NUM_AGENTS = 64        # Instead of 16
    CORE_NUM_QUESTIONS = 16  # Instead of 8
    NUM_QUESTIONS = 8       # Instead of 8
    K_MATCHES = 4
    TOPIC = "artificial intelligence"

    all_agents = Agent.all_from_cluster(TOPIC, NUM_AGENTS)
    random.shuffle(all_agents)

    graph_data = GraphData(all_agents)
    core_agent = all_agents[0]

    total_questions_asked = CORE_NUM_QUESTIONS + (NUM_QUESTIONS*K_MATCHES) * NUM_QUESTIONS

    all_questions = Question.all_from_cluster(TOPIC, total_questions_asked + 100)
    random.shuffle(all_questions)

    assert len(all_questions) >= total_questions_asked, "Not enough questions" 

    if False:
        direct_target_agents = []

        for question in tqdm(all_questions[:CORE_NUM_QUESTIONS], desc="Core agent asking questions"):
            print(f"\n{Fore.GREEN}Core agent asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
            # Ensure question embedding is loaded
            if question.embedding is None:
                question.load_embedding()
                
            matches = agents.match_by_qid(question.id, K_MATCHES, whitelist=[a.id for a in all_agents if a.id != core_agent.id])
            assert len(matches) == K_MATCHES
            for match in matches:
                answer = match.agent.ask(question.question)
                grade, _ = eval.grade_answer(question.question, answer)
                print(f"{core_agent.name} asked {match.agent.name}: {grade}")
                graph_data.add_edge(core_agent, match.agent, question.embedding, grade)
                direct_target_agents.append(match.agent)

        questions_index = CORE_NUM_QUESTIONS

        for agent in tqdm(direct_target_agents, desc="Target agents"):
            for question in tqdm(all_questions[questions_index:questions_index+NUM_QUESTIONS], desc="Agent asking questions"):
                print(f"\n{Fore.GREEN}Agent {agent.name} asking question {Style.BRIGHT}\"{question.question}\"{Style.RESET_ALL}")
                # Ensure question embedding is loaded
                if question.embedding is None:
                    question.load_embedding()
                    
                matches = agents.match_by_qid(question.id, K_MATCHES, whitelist=[a.id for a in all_agents if a.id != agent.id])
                assert len(matches) == K_MATCHES
                for match in matches:
                    answer = match.agent.ask(question.question)
                    grade, _ = eval.grade_answer(question.question, answer)
                    print(f"{agent.name} asked {match.agent.name}: {grade}")
                    graph_data.add_edge(agent, match.agent, question.embedding, grade)
            questions_index += NUM_QUESTIONS
        
        
        with open('data/graph.pkl', 'wb') as f:
            pickle.dump({
                'data': graph_data,
            }, f)
        print(f"\n{Fore.GREEN}Saved graph data to data/graph.pkl{Style.RESET_ALL}")

        # 1. Visualize the initial graph
        visualize_graph(graph_data, title=f"Graph")
        sys.exit()
    else:
        with open('data/graph.pkl', 'rb') as f:
            graph_data: GraphData = pickle.load(f)['data']

    # Instantiate the model
    node_feature_dim = graph_data.x.size(1)
    model = TrustGNN(num_nodes=len(graph_data.agents), node_feature_dim=node_feature_dim)
    evaluate_and_predict(model, graph_data, title="Predictions Before Training")
    
    best_val_loss = train_model(model, graph_data)
    print(f"Training completed with best validation loss: {best_val_loss:.6f}")
    
    evaluate_and_predict(model, graph_data, title="Predictions After Training")

    # Evaluate another question for comparison
    print(f"{Fore.BLUE}{'-'*100}{Style.RESET_ALL}")
    for i in range(100):
        q = all_questions[total_questions_asked+i]
        print(f"\n{Fore.GREEN}Predicting trust scores for top-3 matched agents for question {Style.BRIGHT}\"{q.question}\"{Style.RESET_ALL}")

        k = 3
        source_idx = graph_data.agent_id_to_index[core_agent.id]
        
        # Get top-3 matched agents
        matches = agents.match_by_qid(q.id, k, whitelist=[agent.id for agent in graph_data.agents])
        
        # Get target indices for matched agents
        target_indices = []
        for match in matches[:3]:
            target_idx = graph_data.agent_id_to_index[match.agent.id]
            target_indices.append(target_idx)
        
        # Predict trust scores for matched agents
        predicted_trust_scores = predict_trust_scores(model, graph_data, source_idx, target_indices, q)
        
        # Evaluate and compare actual vs predicted for matched agents
        for idx, match in enumerate(matches[:3]):
            grade, _ = eval.grade_answer(q.question, match.agent.ask(q.question))
            predicted_score = predicted_trust_scores[idx] if idx < len(predicted_trust_scores) else 0.0
            print(f"{Fore.YELLOW}MATCHED {idx+1}:\t{match.agent.name} | Actual: {Fore.GREEN if grade > 0 else Fore.RED}{grade:.2f}{Style.RESET_ALL} | Predicted: {Fore.CYAN}{predicted_score:.4f}{Style.RESET_ALL}")

        print(f"{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")