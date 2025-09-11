import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
from .mhits import calculate_mhits_scores
import os

def evaluate_mhits_with_reranking(graph_file: str, beta: float = 0.3):
    """
    Evaluate MHITS algorithm with graph data.
    Trust is derived implicitly from the rating patterns.
    
    Args:
        graph_file: Path to CSV file with rating edges
        beta: Weight parameter for MHITS algorithm
    
    Returns:
        DataFrame with agent rankings
    """
    # Load graph data (ratings)
    graph_data = pd.read_csv(graph_file, dtype={
        'source_agent': int, 
        'target_agent': int, 
        'question': int, 
        'score': float
    })
    
    # Calculate MHITS scores
    print(f"Calculating MHITS scores with beta={beta}...")
    print(f"Graph has {len(graph_data)} edges and {len(pd.concat([graph_data['source_agent'], graph_data['target_agent']]).unique())} unique agents")
    
    results = calculate_mhits_scores(
        graph_df=graph_data,
        beta=beta
    )
    
    return results

def test_mhits():
    """Test MHITS implementation with synthetic data."""
    # Create synthetic rating data
    # In this data: agents 1-5 are users, agents 10-12 are media/items being rated
    ratings = [
        {'source_agent': 1, 'target_agent': 10, 'question': 1, 'score': 0.8},
        {'source_agent': 1, 'target_agent': 11, 'question': 2, 'score': 0.9},
        {'source_agent': 2, 'target_agent': 10, 'question': 1, 'score': 0.7},
        {'source_agent': 2, 'target_agent': 12, 'question': 3, 'score': 0.6},
        {'source_agent': 3, 'target_agent': 11, 'question': 2, 'score': 0.85},
        {'source_agent': 3, 'target_agent': 12, 'question': 3, 'score': 0.75},
        {'source_agent': 4, 'target_agent': 10, 'question': 1, 'score': 0.9},
        {'source_agent': 5, 'target_agent': 11, 'question': 2, 'score': 0.8},
        # Add some inter-agent ratings to create trust signals
        {'source_agent': 1, 'target_agent': 2, 'question': 4, 'score': 0.8},
        {'source_agent': 2, 'target_agent': 3, 'question': 4, 'score': 0.7},
        {'source_agent': 3, 'target_agent': 1, 'question': 4, 'score': 0.85},
        {'source_agent': 4, 'target_agent': 5, 'question': 4, 'score': 0.9},
    ]
    
    graph_df = pd.DataFrame(ratings)
    
    print("Testing MHITS algorithm with synthetic data...")
    print("\nRating Graph (both media ratings and inter-agent trust):")
    print(graph_df)
    print(f"\nUnique agents: {sorted(pd.concat([graph_df['source_agent'], graph_df['target_agent']]).unique())}")
    
    # Calculate MHITS scores
    results = calculate_mhits_scores(
        graph_df=graph_df,
        beta=0.3
    )
    
    print("\nMHITS Results:")
    print(results)
    print("\nTop 5 Experts (by hubness score):")
    print(results.head())
    
    return results

if __name__ == "__main__":
    for attack_volume in range(0, 101, 10):
        graph_file = f"data/graph/edges_{attack_volume}.csv"
        
        print(f"\n{'='*70}")
        print(f"Testing with attack volume: {attack_volume}%")
        print('='*70)
        
        mhits_results = evaluate_mhits_with_reranking(
            graph_file=graph_file,
            beta=0.3
        )
        
        if mhits_results is not None:
            print(f"\nTop 10 experts for attack_volume={attack_volume}%:")
            print(mhits_results.head(10))