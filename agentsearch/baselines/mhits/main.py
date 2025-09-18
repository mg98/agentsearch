import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from agentsearch.dataset.agents import AgentStore, Agent
from agentsearch.dataset.questions import Question
from .mhits import calculate_mhits_scores
from agentsearch.utils.eval import load_all_attacks, load_test_questions

if __name__ == "__main__":
    for attack_volume, graph_df in load_all_attacks():
        print(f"\n{'='*70}")
        print(f"Testing with attack volume: {attack_volume}%")
        print('='*70)
        
        mhits_scores = calculate_mhits_scores(graph_df, beta=0.3)
        print(mhits_scores.head(10))
        
        test_questions = load_test_questions()