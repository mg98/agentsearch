import os
import numpy as np
import pandas as pd
from colorama import init, Fore, Style
from agentsearch.dataset.agents import Agent
from agentsearch.dataset.questions import Question
from agentsearch.graph.utils import compute_trust_score

# Initialize colorama
init(autoreset=True)

def evaluate_agent_card_matching():
    """Evaluate top-1 agent card matching for questions from test_qids.txt"""
    
    # Check if test_qids.txt exists
    test_qids_path = 'data/test_qids.txt'
    if not os.path.exists(test_qids_path):
        print(f"{Fore.RED}Error: {test_qids_path} file not found{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please create a comma-separated list of question IDs in {test_qids_path}{Style.RESET_ALL}")
        return
    
    # Load question IDs from file
    with open(test_qids_path, 'r') as f:
        content = f.read().strip()
        if not content:
            print(f"{Fore.RED}Error: {test_qids_path} is empty{Style.RESET_ALL}")
            return
        test_qids = [int(qid.strip()) for qid in content.split(',')]
    
    print(f"{Fore.GREEN}Loaded {len(test_qids)} question IDs from {test_qids_path}{Style.RESET_ALL}")
    
    # Get all available agents
    all_agents = Agent.all(collection="agents")
    agent_ids = [agent.id for agent in all_agents]
    print(f"{Fore.CYAN}Found {len(agent_ids)} agents in the database{Style.RESET_ALL}")
    
    # Track performance metrics
    scores = []
    
    print(f"\n{Fore.BLUE}{'='*100}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Starting Top-1 Agent Card Matching Evaluation{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*100}{Style.RESET_ALL}")
    
    # Evaluate each question
    for i, qid in enumerate(test_qids):
        # Load the question
        question = Question.from_id(qid)
        print(f"\n{Fore.GREEN}Question {i+1}/{len(test_qids)} (ID: {qid}):{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}\"{question.text[:100]}{'...' if len(question.text) > 100 else ''}\"{Style.RESET_ALL}")

        # Get top-1 matched agent
        print(qid)
        matches = Agent.match(question, top_k=1, collection="agents")
        matched_agent = matches[0].agent
        score = compute_trust_score(matched_agent.count_sources(question.text))
        scores.append(score) 
        
        print(f"{Fore.YELLOW}Matched Agent:{Style.RESET_ALL} {matched_agent.name}")
        print(f"{Fore.YELLOW}Answer Quality Grade:{Style.RESET_ALL} {Fore.GREEN if score > 0.5 else Fore.RED}{score:.3f}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")
    
    # Convert to numpy array for analysis
    scores = np.array(scores)
    
    # Print performance summary
    print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PERFORMANCE SUMMARY - TOP-1 AGENT CARD MATCHING{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    
    print(f"\n{Fore.GREEN}Overall Statistics:{Style.RESET_ALL}")
    print(f"  Total Questions: {len(scores)}")
    print(f"  Mean Grade: {scores.mean():.4f}")
    print(f"  Median Grade: {np.median(scores):.4f}")
    print(f"  Std Dev: {scores.std():.4f}")
    print(f"  Min Grade: {scores.min():.4f}")
    print(f"  Max Grade: {scores.max():.4f}")
    print(f"  25th Percentile: {np.percentile(scores, 25):.4f}")
    print(f"  75th Percentile: {np.percentile(scores, 75):.4f}")
    
    # Count successful vs unsuccessful matches
    successful = np.sum(scores > 0)
    print(f"\n{Fore.GREEN}Success Rate:{Style.RESET_ALL}")
    print(f"  Successful Answers: {successful}/{len(scores)} ({100*successful/len(scores):.1f}%)")
    print(f"  Failed Answers: {len(scores)-successful}/{len(scores)} ({100*(len(scores)-successful)/len(scores):.1f}%)")
    
    # Save summary statistics
    summary_stats = {
        'metric': ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'success_rate', 'sample_size'],
        'value': [
            scores.mean(),
            np.median(scores),
            scores.std(),
            scores.min(),
            scores.max(),
            np.percentile(scores, 25),
            np.percentile(scores, 75),
            successful/len(scores),
            len(scores)
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_file = 'agentcard_matching_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"{Fore.CYAN}Summary statistics saved to:{Style.RESET_ALL} {summary_file}")

if __name__ == '__main__':
    evaluate_agent_card_matching()