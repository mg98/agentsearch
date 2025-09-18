import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import torch
from .nerank import NeRank


def nerank_search(
    agents_df: pd.DataFrame,
    agent_id: str,
    question: str,
    paper_id: str,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Perform NeRank-based personalized question routing.
    
    Args:
        agents_df: DataFrame containing agent information
        agent_id: ID of the agent asking the question (raiser)
        question: Question text
        paper_id: Paper ID for context
        config: Optional configuration dictionary
        
    Returns:
        DataFrame with agents ranked by their scores
    """
    # Default configuration
    default_config = {
        'embedding_dim': 128,  # Reduced for efficiency
        'lstm_hidden_dim': 128,
        'lstm_layers': 1,
        'cnn_num_filters': 32,
        'cnn_hidden_dim': 64,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 16,
        'num_negative': 3,
        'metapath': 'AQRQA',
        'walk_length': 9,
        'walks_per_node': 10,
        'max_question_length': 50,
        'embedding_epochs': 5,
        'ranking_epochs': 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if config:
        default_config.update(config)
    
    # Build graph data from agents_df
    graph_data = _build_graph_data(agents_df)
    
    # Build question texts (simplified - using question as both ID and text)
    question_texts = {question: question}
    
    # Add historical questions if available
    if 'past_questions' in agents_df.columns:
        for _, agent in agents_df.iterrows():
            if pd.notna(agent.get('past_questions')):
                for q in str(agent['past_questions']).split(';'):
                    if q.strip():
                        question_texts[q.strip()] = q.strip()
    
    # Initialize NeRank model
    model = NeRank(default_config)
    model.setup(graph_data, question_texts)
    
    # Train embeddings (simplified training for efficiency)
    if len(graph_data.get('raiser_questions', {})) > 0:
        model.train_embeddings(num_epochs=default_config['embedding_epochs'])
        
        # Create simplified training data
        training_data = _create_training_data(graph_data, agents_df)
        if training_data:
            model.train_ranking(training_data, num_epochs=default_config['ranking_epochs'])
    
    # Get candidate answerers
    candidates = agents_df['agent_id'].tolist()
    
    # Rank answerers
    rankings = model.rank_answerers(agent_id, question, candidates)
    
    # Create result DataFrame
    result_df = pd.DataFrame(rankings, columns=['agent_id', 'nerank_score'])
    
    # Merge with original agent information
    result_df = result_df.merge(agents_df, on='agent_id', how='left')
    
    # Sort by score
    result_df = result_df.sort_values('nerank_score', ascending=False)
    
    return result_df


def _build_graph_data(agents_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build graph data from agents DataFrame.
    
    Args:
        agents_df: DataFrame containing agent information
        
    Returns:
        Dictionary with graph relationships
    """
    graph_data = {
        'raiser_questions': {},
        'answerer_questions': {}
    }
    
    # Build relationships from agent interactions
    for _, agent in agents_df.iterrows():
        agent_id = agent['agent_id']
        
        # Simulate raiser-question relationships based on agent expertise
        if 'expertise' in agent and pd.notna(agent['expertise']):
            # Agents with expertise can ask questions in their domain
            questions = [f"q_{agent_id}_{i}" for i in range(3)]
            graph_data['raiser_questions'][f"r_{agent_id}"] = questions
            
            # Same agents can answer questions
            graph_data['answerer_questions'][f"a_{agent_id}"] = questions
    
    # Add some cross-connections for richer graph structure
    all_questions = []
    for questions in graph_data['raiser_questions'].values():
        all_questions.extend(questions)
    
    # Each answerer can potentially answer multiple questions
    for answerer in list(graph_data['answerer_questions'].keys())[:5]:
        if all_questions:
            # Add connections to some random questions
            additional_questions = np.random.choice(
                all_questions, 
                size=min(3, len(all_questions)), 
                replace=False
            ).tolist()
            graph_data['answerer_questions'][answerer].extend(additional_questions)
    
    return graph_data


def _create_training_data(graph_data: Dict[str, Any], agents_df: pd.DataFrame) -> List[Dict]:
    """
    Create simplified training data for ranking.
    
    Args:
        graph_data: Graph relationships
        agents_df: DataFrame containing agent information
        
    Returns:
        List of training samples
    """
    training_data = []
    
    # Get all answerers
    all_answerers = list(graph_data['answerer_questions'].keys())
    
    # Create training samples
    for raiser, questions in graph_data['raiser_questions'].items():
        for question in questions:
            # Find answerers for this question
            answerers_for_q = [
                answerer for answerer, qs in graph_data['answerer_questions'].items()
                if question in qs
            ]
            
            if len(answerers_for_q) >= 2:
                # Simulate accepted answerer (first one)
                accepted = answerers_for_q[0]
                others = answerers_for_q[1:]
                
                # Get unanswered (answerers not connected to this question)
                unanswered = [a for a in all_answerers if a not in answerers_for_q]
                
                if unanswered:
                    training_data.append({
                        'raiser': raiser,
                        'question': question,
                        'accepted_answerer': accepted,
                        'other_answerers': others,
                        'unanswered': unanswered[:5]  # Sample a few
                    })
    
    return training_data