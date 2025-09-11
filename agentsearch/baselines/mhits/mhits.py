import pandas as pd
import numpy as np

def calculate_mhits_scores(graph_df: pd.DataFrame, 
                           beta: float = 0.3, max_iterations: int = 100, 
                           convergence_threshold: float = 1e-6) -> pd.DataFrame:
    """
    Calculate MHITS hub and authority scores iteratively.
    
    Based on formulas from Rashed et al. 2012:
    - a(m) = Σ h(u) * r(u) for all users u who rated media m
    - h(u) = β * Σ a(m) * r(u) + (1-β) * t(u) for all media m rated by user u
    
    Trust is derived from the graph itself - agents who rate similarly
    or interact positively have implicit trust.
    
    Args:
        graph_df: DataFrame with columns ['source_agent', 'target_agent', 'score']
                 where source_agent rates target_agent with score
        beta: Weight parameter for balancing authority and trust components
        max_iterations: Maximum number of iterations
        convergence_threshold: Convergence threshold for stopping
    
    Returns:
        DataFrame with agent_id, hubness_score, and authority_score
    """
    # Get all unique agent IDs
    all_agent_ids = pd.concat([
        graph_df['source_agent'], 
        graph_df['target_agent']
    ]).unique()
    
    # Initialize scores
    n_agents = len(all_agent_ids)
    hubness_scores = {agent_id: 1.0/n_agents for agent_id in all_agent_ids}
    authority_scores = {agent_id: 1.0/n_agents for agent_id in all_agent_ids}
    
    # Derive trust from the graph - trust is based on incoming positive ratings
    # Agents who receive high scores from others are considered more trustworthy
    trust_averages = {}
    for agent_id in all_agent_ids:
        # Trust is based on incoming positive ratings (how others rate this agent)
        incoming_ratings = graph_df[graph_df['target_agent'] == agent_id]['score'].values
        trust_averages[agent_id] = np.mean(incoming_ratings)
    
    # Iterative calculation
    for iteration in range(max_iterations):
        prev_hubness = hubness_scores.copy()
        prev_authority = authority_scores.copy()
        
        # Update authority scores: a(m) = Σ h(u) * r(u)
        new_authority_scores = {}
        for agent_id in all_agent_ids:
            # Get all ratings for this agent (as target)
            ratings_for_agent = graph_df[graph_df['target_agent'] == agent_id]
            
            authority_sum = 0.0
            for _, row in ratings_for_agent.iterrows():
                source_agent = row['source_agent']
                rating = row['score']
                if source_agent in hubness_scores:
                    authority_sum += hubness_scores[source_agent] * rating
            
            new_authority_scores[agent_id] = authority_sum
        
        # Update hub scores: h(u) = β * Σ a(m) * r(u) + (1-β) * t(u)
        new_hubness_scores = {}
        for agent_id in all_agent_ids:
            # Get all ratings by this agent (as source)
            ratings_by_agent = graph_df[graph_df['source_agent'] == agent_id]
            
            authority_part = 0.0
            for _, row in ratings_by_agent.iterrows():
                target_agent = row['target_agent']
                rating = row['score']
                if target_agent in new_authority_scores:
                    authority_part += new_authority_scores[target_agent] * rating
            
            trust_part = trust_averages.get(agent_id, 0.0)
            new_hubness_scores[agent_id] = beta * authority_part + (1 - beta) * trust_part
        
        # Normalize scores
        if sum(new_authority_scores.values()) > 0:
            auth_sum = sum(new_authority_scores.values())
            new_authority_scores = {k: v/auth_sum for k, v in new_authority_scores.items()}
        
        if sum(new_hubness_scores.values()) > 0:
            hub_sum = sum(new_hubness_scores.values())
            new_hubness_scores = {k: v/hub_sum for k, v in new_hubness_scores.items()}
        
        # Update scores
        authority_scores = new_authority_scores
        hubness_scores = new_hubness_scores
        
        # Check convergence
        hub_diff = sum(abs(hubness_scores[aid] - prev_hubness[aid]) 
                      for aid in all_agent_ids)
        auth_diff = sum(abs(authority_scores[aid] - prev_authority[aid]) 
                       for aid in all_agent_ids)
        
        if hub_diff < convergence_threshold and auth_diff < convergence_threshold:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    # Create results DataFrame
    results = pd.DataFrame({
        'agent_id': list(all_agent_ids),
        'hubness_score': [hubness_scores[aid] for aid in all_agent_ids],
        'authority_score': [authority_scores[aid] for aid in all_agent_ids]
    })
    
    # Sort by hubness score (expert ranking)
    results = results.sort_values('hubness_score', ascending=False)
    
    return results