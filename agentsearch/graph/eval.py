import torch
import random
import pickle
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.graph.training import train_model, evaluate_and_predict, predict_top_targets, predict_trust_scores
from tqdm import tqdm
import math

# Initialize colorama
init(autoreset=True)

random.seed(42)

if __name__ == '__main__':
    with open('data/graph.pkl', 'rb') as f:
        data = pickle.load(f)
        graph_data: GraphData = data['graph']
        graph_data.finalize_features()
        all_questions: list[Question] = data['questions']
        total_questions_asked: int = data['total_questions_asked']
        core_agent: Agent = data['core_agent']

    # Print trust bounds for debugging
    print(f"Original trust bounds: min={graph_data.original_trust_min}, max={graph_data.original_trust_max}")
    
    # Instantiate the model
    node_feature_dim = graph_data.x.size(1)
    model = TrustGNN(num_nodes=len(graph_data.agents), node_feature_dim=node_feature_dim)
    evaluate_and_predict(model, graph_data, title="Predictions Before Training")
    
    best_val_loss = train_model(model, graph_data)
    print(f"Training completed with best validation loss: {best_val_loss:.6f}")
    
    evaluate_and_predict(model, graph_data, title="Predictions After Training")
    # sys.exit()

    # Evaluate another question for comparison
    print(f"{Fore.BLUE}{'-'*100}{Style.RESET_ALL}")
    for i in range(100):
        q = all_questions[total_questions_asked+i]
        print(f"\n{Fore.GREEN}Predicting trust scores for top-3 matched agents for question {Style.BRIGHT}\"{q.question}\"{Style.RESET_ALL}")

        k = 3
        source_idx = graph_data.agent_id_to_index[core_agent.id]

        # Create prediction data for all possible target agents
        all_target_indices = [i for i in range(len(graph_data.agents)) if i != source_idx]
        
        # Predict trust scores using GNN
        all_predicted_scores = predict_trust_scores(model, graph_data, source_idx, all_target_indices, q)
        
        # Get indices of top-3 predicted agents by GNN
        top_3_indices_gnn = torch.argsort(torch.tensor(all_predicted_scores), descending=True)[:3]
        
        print(f"\n{Fore.GREEN}Top 3 GNN predictions:{Style.RESET_ALL}")
        for i, idx in enumerate(top_3_indices_gnn):
            agent = graph_data.agents[all_target_indices[idx]]
            predicted_score = all_predicted_scores[idx]
            raw_grade = agent.ask(q.question)
            # Use the proper normalization method from GraphData
            grade = graph_data.normalize_single_score(raw_grade)
            
            print(f"{Fore.YELLOW}GNN {i+1}:\t{agent.name} | Raw: {Fore.LIGHTBLACK_EX}{raw_grade}{Style.RESET_ALL} | Actual: {Fore.GREEN if grade > 0 else Fore.RED}{grade:.2f}{Style.RESET_ALL} | Predicted: {Fore.CYAN}{predicted_score:.4f}{Style.RESET_ALL}")
        
        print(f"{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")
        
        # Get top-3 matched agents (original matching algorithm)
        matches = agents.match_by_qid(q.id, k, whitelist=[agent.id for agent in graph_data.agents])
        
        # Get target indices for matched agents
        target_indices = []
        for match in matches[:3]:
            target_idx = graph_data.agent_id_to_index[match.agent.id]
            target_indices.append(target_idx)
        
        # Predict trust scores for matched agents using GNN
        predicted_trust_scores = predict_trust_scores(model, graph_data, source_idx, target_indices, q)
        
        # Evaluate and compare actual vs predicted for matched agents
        for idx, match in enumerate(matches[:3]):
            raw_grade = match.agent.ask(q.question)
            # Use the proper normalization method from GraphData
            grade = graph_data.normalize_single_score(raw_grade)
            
            predicted_score = predicted_trust_scores[idx] if idx < len(predicted_trust_scores) else 0.0
            print(f"{Fore.YELLOW}MATCHED {idx+1}:\t{match.agent.name} | Raw: {Fore.LIGHTBLACK_EX}{raw_grade}{Style.RESET_ALL} | Actual: {Fore.GREEN if grade > 0 else Fore.RED}{grade:.2f}{Style.RESET_ALL} | Predicted: {Fore.CYAN}{predicted_score:.4f}{Style.RESET_ALL}")

        print(f"{Fore.BLUE}{'-' * 50}{Style.RESET_ALL}")