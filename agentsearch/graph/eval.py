import torch
import random
import pickle
import os
import numpy as np
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.dataset import agents
from agentsearch.graph.types import GraphData, TrustGNN
from agentsearch.graph.training import train_model, evaluate_and_predict, predict_top_targets, predict_trust_scores

# Initialize colorama
init(autoreset=True)

random.seed(42)

if __name__ == '__main__':
    if not os.path.exists('data/graph.pkl'):
        print("Error: data/graph.pkl file not found")
        exit(1)
    
    with open('data/graph.pkl', 'rb') as f:
        data = pickle.load(f)
        graph_data: GraphData = data['graph']
        graph_data.finalize_features()
        core_agent: Agent = data['core_agent']
        test_questions: list[Question] = data['test_questions']

    # Print trust bounds for debugging
    print(f"Original trust bounds: min={graph_data.original_trust_min}, max={graph_data.original_trust_max}")
    
    # Instantiate the model
    node_feature_dim = graph_data.x.size(1)
    model = TrustGNN(num_nodes=len(graph_data.agents), node_feature_dim=node_feature_dim)
    # evaluate_and_predict(model, graph_data, title="Predictions Before Training")
    
    best_val_loss = train_model(model, graph_data)
    print(f"Training completed with best validation loss: {best_val_loss:.6f}")
    
    evaluate_and_predict(model, graph_data, title="Predictions After Training")

    # Track top-1 performance data
    gnn_predicted_scores = []
    gnn_actual_grades = []
    matched_predicted_scores = []
    matched_actual_grades = []
    
    # Evaluate questions and track top-1 performance
    print(f"{Fore.BLUE}{'-'*100}{Style.RESET_ALL}")
    num_questions = len(test_questions)
    
    for i, q in enumerate(test_questions):
        print(f"\n{Fore.GREEN}Question {i+1}/{num_questions}: {Style.BRIGHT}\"{q.question[:100]}...\"{Style.RESET_ALL}")

        source_idx = graph_data.agent_id_to_index[core_agent.id]

        # Create prediction data for all possible target agents
        all_target_indices = [j for j in range(len(graph_data.agents)) if j != source_idx]
        
        # === GNN TOP-1 PREDICTION ===
        # Predict trust scores using GNN
        all_predicted_scores = predict_trust_scores(model, graph_data, source_idx, all_target_indices, q)
        
        # Get top-1 predicted agent by GNN
        top_1_idx_gnn = torch.argsort(torch.tensor(all_predicted_scores), descending=True)[0]
        
        gnn_agent = graph_data.agents[all_target_indices[top_1_idx_gnn]]
        gnn_predicted_score = all_predicted_scores[top_1_idx_gnn]
        gnn_raw_grade = gnn_agent.ask(q.question)
        gnn_actual_grade = graph_data.normalize_single_score(gnn_raw_grade)
        
        # Store GNN top-1 results
        gnn_predicted_scores.append(gnn_predicted_score)
        gnn_actual_grades.append(gnn_actual_grade)
        
        print(f"{Fore.YELLOW}GNN TOP-1:\t{gnn_agent.name} | Actual: {Fore.GREEN if gnn_actual_grade > 0 else Fore.RED}{gnn_actual_grade:.3f}{Style.RESET_ALL} | Predicted: {Fore.CYAN}{gnn_predicted_score:.3f}{Style.RESET_ALL}")
        
        # === MATCHED TOP-1 AGENT ===
        # Get top-1 matched agent (original matching algorithm)
        matches = agents.match_by_qid(q.id, 1, whitelist=[agent.id for agent in graph_data.agents])
        
        if matches:
            matched_agent = matches[0].agent
            matched_target_idx = graph_data.agent_id_to_index[matched_agent.id]
            
            # Predict trust score for matched agent using GNN
            matched_predicted_score = predict_trust_scores(model, graph_data, source_idx, [matched_target_idx], q)[0]
            matched_raw_grade = matched_agent.ask(q.question)
            matched_actual_grade = graph_data.normalize_single_score(matched_raw_grade)
            
            # Store matched top-1 results
            matched_predicted_scores.append(matched_predicted_score)
            matched_actual_grades.append(matched_actual_grade)
            
            print(f"{Fore.YELLOW}MATCHED TOP-1:\t{matched_agent.name} | Actual: {Fore.GREEN if matched_actual_grade > 0 else Fore.RED}{matched_actual_grade:.3f}{Style.RESET_ALL} | Predicted: {Fore.CYAN}{matched_predicted_score:.3f}{Style.RESET_ALL}")
        else:
            # Handle case where no matches found
            matched_predicted_scores.append(0.5)
            matched_actual_grades.append(0.5)
            print(f"{Fore.RED}No matched agents found{Style.RESET_ALL}")

        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")

    # Convert to numpy arrays for analysis
    gnn_predicted = np.array(gnn_predicted_scores)
    gnn_actual = np.array(gnn_actual_grades)
    matched_predicted = np.array(matched_predicted_scores)
    matched_actual = np.array(matched_actual_grades)
    
    # Print performance summary
    print(f"\n{Fore.CYAN}=== PERFORMANCE SUMMARY ==={Style.RESET_ALL}")
    
    # Calculate metrics
    gnn_mse = np.mean((gnn_predicted - gnn_actual) ** 2)
    matched_mse = np.mean((matched_predicted - matched_actual) ** 2)
    gnn_mae = np.mean(np.abs(gnn_predicted - gnn_actual))
    matched_mae = np.mean(np.abs(matched_predicted - matched_actual))
    gnn_corr = np.corrcoef(gnn_predicted, gnn_actual)[0, 1]
    matched_corr = np.corrcoef(matched_predicted, matched_actual)[0, 1]
    
    print(f"{Fore.GREEN}GNN Top-1 Performance:{Style.RESET_ALL}")
    print(f"  MSE: {gnn_mse:.4f}")
    print(f"  MAE: {gnn_mae:.4f}")
    print(f"  Correlation: {gnn_corr:.4f}")
    print(f"  Mean Actual Grade: {gnn_actual.mean():.3f}")
    print(f"  Mean Predicted Score: {gnn_predicted.mean():.3f}")
    
    print(f"\n{Fore.GREEN}Matched Top-1 Performance:{Style.RESET_ALL}")
    print(f"  MSE: {matched_mse:.4f}")
    print(f"  MAE: {matched_mae:.4f}")
    print(f"  Correlation: {matched_corr:.4f}")
    print(f"  Mean Actual Grade: {matched_actual.mean():.3f}")
    print(f"  Mean Predicted Score: {matched_predicted.mean():.3f}")
    
    # Save data for R visualization
    import pandas as pd
    
    # Prepare data for R
    # Create a long-format dataframe for easier plotting in R
    data_for_r = []
    
    # Add GNN data
    for grade in gnn_actual:
        data_for_r.append({
            'method': 'GNN_Top1',
            'actual_grade': grade
        })
    
    # Add Matched data
    for grade in matched_actual:
        data_for_r.append({
            'method': 'Matched_Top1', 
            'actual_grade': grade
        })
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data_for_r)
    df.to_csv('top1_grades_data.csv', index=False)
    
    # Also save summary statistics
    summary_stats = {
        'method': ['GNN_Top1', 'Matched_Top1'],
        'mean': [gnn_actual.mean(), matched_actual.mean()],
        'median': [np.median(gnn_actual), np.median(matched_actual)],
        'std': [gnn_actual.std(), matched_actual.std()],
        'min': [gnn_actual.min(), matched_actual.min()],
        'max': [gnn_actual.max(), matched_actual.max()],
        'q25': [np.percentile(gnn_actual, 25), np.percentile(matched_actual, 25)],
        'q75': [np.percentile(gnn_actual, 75), np.percentile(matched_actual, 75)],
        'sample_size': [len(gnn_actual), len(matched_actual)]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('top1_grades_summary.csv', index=False)
    
    print(f"\n{Fore.CYAN}Data saved to:{Style.RESET_ALL}")
    print(f"  - top1_grades_data.csv (raw data for plotting)")
    print(f"  - top1_grades_summary.csv (summary statistics)")
    print(f"\n{Fore.GREEN}Sample sizes:{Style.RESET_ALL}")
    print(f"  - GNN Top-1: {len(gnn_actual)} questions")
    print(f"  - Matched Top-1: {len(matched_actual)} questions")