import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
import glob
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from agentsearch.forc.model import FORCMetaModel, FORCTrainer, ModelConfig
from agentsearch.dataset.questions import Question, questions_df
from agentsearch.dataset.agents import AgentStore


def load_test_qids(filepath='data/test_qids.txt'):
    """Load test question IDs from file."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    test_qids = [int(qid) for qid in content.split(',')]
    return test_qids


def prepare_training_data(graph_path, classification_threshold=0.5):
    """Load and prepare training data from graph CSV, converting scores to binary labels."""

    print(f"Loading training data from {graph_path}...")
    df = pd.read_csv(graph_path)
    valid_questions = set(questions_df.index)
    df = df[df['question'].isin(valid_questions)]
    print(f"  Loaded {len(df)} training samples")
    
    queries = []
    agent_ids = []
    targets = []
    
    for _, row in df.iterrows():
        question_id = int(row['question'])
        agent_id = int(row['target_agent'])
        score = float(row['score'])
        
        question_text = questions_df.loc[question_id, 'question']
        queries.append(question_text)
        agent_ids.append(agent_id)
        binary_label = 1.0 if score >= classification_threshold else 0.0
        targets.append(binary_label)
    
    # Print class distribution
    positive_samples = sum(1 for t in targets if t == 1.0)
    negative_samples = len(targets) - positive_samples
    print(f"  Class distribution: {positive_samples} positive ({positive_samples/len(targets)*100:.1f}%), "
          f"{negative_samples} negative ({negative_samples/len(targets)*100:.1f}%)")
    
    return queries, agent_ids, targets, df


def train_meta_model(train_data, graph_id, epochs=5):
    """Train the FORC meta-model."""
    queries, agent_ids, targets, df = train_data
    
    # Split into train/val  
    train_queries, val_queries, train_agents, val_agents, train_targets, val_targets = train_test_split(
        queries, agent_ids, targets, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    train_targets = torch.tensor(train_targets, dtype=torch.float32).reshape(-1, 1)
    val_targets = torch.tensor(val_targets, dtype=torch.float32).reshape(-1, 1)
    
    # Model configuration
    config = ModelConfig(
        learning_rate=1e-4,
        max_length=256,
        classification_threshold=0.5
    )
    
    model = FORCMetaModel(config)
    trainer = FORCTrainer(model, device='mps')
    print(f"  Training on {len(train_queries)} samples, validating on {len(val_queries)}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_losses = []
        batch_size = 64
        
        # Create progress bar for training batches
        num_batches = (len(train_queries) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(train_queries), batch_size), 
                    desc=f"  Epoch {epoch+1}/{epochs} - Training",
                    total=num_batches)
        
        for i in pbar:
            batch_queries = train_queries[i:i+batch_size]
            batch_agents = train_agents[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            # Pack data into tuples for train_step
            training_data = list(zip(batch_queries, batch_agents, 
                                    batch_targets.squeeze().tolist()))
            
            loss = trainer.train_step(training_data)
            epoch_losses.append(loss)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = np.mean(epoch_losses)
        
        # Validation - pack data into tuples for evaluate
        val_data = list(zip(val_queries, val_agents, val_targets.squeeze().tolist()))
        val_metrics = trainer.evaluate(val_data)
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1']
        
        print(f"    Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_accuracy:.4f}, val_f1={val_f1:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }, f'agentsearch/forc/best_model_graph_{graph_id}.pt')
    
    return model, trainer


def evaluate_with_trust_scores(model: FORCMetaModel, trainer: FORCTrainer, test_qids: list[int]):
    """
    Proper evaluation: 
    1. For each test question, predict scores for all agents
    2. Select agent with highest score
    3. Compute trust score using agent.count_sources(question)
    """
    print(f"\\nEvaluating on {len(test_qids)} test questions...")
    
    # Initialize agent store
    agent_store = AgentStore(use_llm_agent_card=False)
    available_agents = agent_store.all(shallow=True)  # Load without papers for speed
    
    trust_scores = []
    predicted_probabilities = []  # Store predicted probabilities for analysis
    model.eval()
    
    with torch.no_grad():
        for qid in tqdm(test_qids, desc="Evaluating test questions"):
            if qid not in questions_df.index:
                continue
                
            question = Question.from_id(qid)
            question_text = question.question
            
            # Predict scores for all agents
            agent_predictions = []
            batch_size = 128
            
            for i in range(0, len(available_agents), batch_size):
                batch_agents = available_agents[i:i+batch_size]
                batch_queries = [question_text] * len(batch_agents)
                batch_agent_ids = [agent.id for agent in batch_agents]
                
                # Get model predictions (probabilities)
                encoded = model.prepare_input(batch_queries, batch_agent_ids)
                input_ids = encoded['input_ids'].to(trainer.device)
                attention_mask = encoded['attention_mask'].to(trainer.device)
                
                predictions = model(input_ids, attention_mask)
                
                # Convert to CPU and extract probabilities
                pred_probs = predictions.cpu().detach().numpy().flatten()
                
                for agent, pred_prob in zip(batch_agents, pred_probs):
                    agent_predictions.append((agent, pred_prob))
            
            # Select agent with highest predicted probability
            best_agent, best_pred_prob = max(agent_predictions, key=lambda x: x[1])
            predicted_probabilities.append(best_pred_prob)
            
            # Compute actual trust score using count_sources
            actual_trust_score = int(best_agent.has_sources(question_text))
            
            trust_scores.append(actual_trust_score)
            
            if len(trust_scores) % 10 == 0:
                print(f"    Processed {len(trust_scores)} questions, avg trust score: {np.mean(trust_scores):.4f}, "
                      f"avg pred prob: {np.mean(predicted_probabilities):.4f}")
    
    return trust_scores, predicted_probabilities


def run_single_experiment(graph_path, test_qids):
    """Run a single training and evaluation experiment."""
    # Extract graph ID from path
    basename = os.path.basename(graph_path)
    graph_id = int(basename.replace('graph_', '').replace('.csv', ''))
    
    print(f"\\n{'='*60}")
    print(f"EXPERIMENT: GRAPH_{graph_id} (Binary Classification)")
    print(f"{'='*60}")
    
    train_data = prepare_training_data(graph_path)
    model, trainer = train_meta_model(train_data, graph_id)
    trust_scores, predicted_probabilities = evaluate_with_trust_scores(model, trainer, test_qids)
    
    result = {
        'graph_id': graph_id,
        'num_training_samples': len(train_data[0]),
        'trust_scores': trust_scores,
        'predicted_probabilities': predicted_probabilities,
        'mean_trust_score': np.mean(trust_scores),
        'std_trust_score': np.std(trust_scores),
        'median_trust_score': np.median(trust_scores),
        'mean_predicted_prob': np.mean(predicted_probabilities),
        'std_predicted_prob': np.std(predicted_probabilities)
    }
    
    print(f"  Results: mean_trust={result['mean_trust_score']:.4f}, "
          f"std_trust={result['std_trust_score']:.4f}, "
          f"mean_pred_prob={result['mean_predicted_prob']:.4f}, "
          f"n_questions={len(trust_scores)}")
    
    return result


def create_boxplot(results):
    """Create boxplot of trust scores across different graph datasets."""
    plt.figure(figsize=(14, 8))
    
    # Prepare data for boxplot
    graph_ids = [r['graph_id'] for r in results]
    trust_scores_data = [r['trust_scores'] for r in results]
    training_samples = [r['num_training_samples'] for r in results]
    
    # Create boxplot with attack volume percentages as labels
    attack_volume_labels = [f'{graph_id}%' for graph_id in graph_ids]
    bp = plt.boxplot(trust_scores_data, tick_labels=attack_volume_labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Attack Volume Percentage', fontsize=12)
    plt.ylabel('Trust Scores (Actual Performance)', fontsize=12)
    plt.title('FORC Meta-Model (Binary Classification): Trust Scores vs Attack Volume\\n' + 
              'Models trained at different attack volumes (0%-100%), evaluated on test questions', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, (graph_id, n_samples) in enumerate(zip(graph_ids, training_samples)):
        plt.text(i+1, 0.95, f'n={n_samples}', 
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('agentsearch/forc/trust_score_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\nBoxplot saved as 'agentsearch/forc/trust_score_boxplot.png'")


def create_prediction_analysis(results):
    """Create additional visualizations for binary classification analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Predicted probabilities vs Trust scores
    graph_ids = [r['graph_id'] for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        trust_scores = result['trust_scores']
        pred_probs = result['predicted_probabilities']
        ax1.scatter(pred_probs, trust_scores, alpha=0.6, color=colors[i], 
                   label=f'Graph {result["graph_id"]}%', s=20)
    
    ax1.set_xlabel('Predicted Probability (Model Output)')
    ax1.set_ylabel('Actual Trust Score')
    ax1.set_title('Predicted Probability vs Actual Trust Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Mean predicted probabilities by attack volume
    mean_probs = [r['mean_predicted_prob'] for r in results]
    std_probs = [r['std_predicted_prob'] for r in results]
    
    ax2.errorbar(graph_ids, mean_probs, yerr=std_probs, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax2.set_xlabel('Attack Volume Percentage')
    ax2.set_ylabel('Mean Predicted Probability')
    ax2.set_title('Model Confidence vs Attack Volume')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('agentsearch/forc/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\nPrediction analysis saved as 'agentsearch/forc/prediction_analysis.png'")


def main():
    """Main evaluation pipeline."""
    print("BINARY CLASSIFICATION FORC EVALUATION PIPELINE")
    print("="*50)
    print("- Training: Use graph_X.csv, convert scores to binary labels (threshold=0.5)")
    print("- Testing: Predict best agent for test_qids.txt, compute trust scores")
    print("="*50)
    
    test_qids = load_test_qids()
    print(f"Loaded {len(test_qids)} test questions")
    
    # Try both possible locations for graph files
    graph_files = sorted(glob.glob('data/graph_*.csv'))
    if not graph_files:
        graph_files = sorted(glob.glob('data/graph/graph_*.csv'))
    print(f"Processing {len(graph_files)} graph files: {[os.path.basename(f) for f in graph_files]}")
    
    results = []
    for graph_path in tqdm(graph_files, desc="Running experiments", unit="graph"):
        result = run_single_experiment(graph_path, test_qids)
        results.append(result)
    
    # Save results
    with open('agentsearch/forc/correct_evaluation_results.json', 'w') as f:
        json_results = []
        for r in results:
            json_r = r.copy()
            # Convert all NumPy arrays to Python lists with float values
            json_r['trust_scores'] = [float(x) for x in json_r['trust_scores']]
            json_r['predicted_probabilities'] = [float(x) for x in json_r['predicted_probabilities']]
            # Convert NumPy scalar values to Python floats
            json_r['mean_trust_score'] = float(json_r['mean_trust_score'])
            json_r['std_trust_score'] = float(json_r['std_trust_score'])
            json_r['median_trust_score'] = float(json_r['median_trust_score'])
            json_r['mean_predicted_prob'] = float(json_r['mean_predicted_prob'])
            json_r['std_predicted_prob'] = float(json_r['std_predicted_prob'])
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    # Create visualizations
    if results:
        create_boxplot(results)
        create_prediction_analysis(results)
        
        # Print summary
        print("\\n" + "="*60)
        print("BINARY CLASSIFICATION EVALUATION SUMMARY")
        print("="*60)
        for result in results:
            print(f"Graph {result['graph_id']:3d}: "
                  f"mean_trust={result['mean_trust_score']:.4f}, "
                  f"std_trust={result['std_trust_score']:.4f}, "
                  f"mean_prob={result['mean_predicted_prob']:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()