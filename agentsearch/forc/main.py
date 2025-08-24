import pandas as pd
from agentsearch.forc.model import FORCMetaModel, ModelConfig, FORCTrainer
from agentsearch.dataset.questions import questions_df
from agentsearch.dataset.agents import agents_df, Agent, AgentStore
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

NUM_EPOCHS = 5
BATCH_SIZE = 128
Data = list[str, int, float] # Question text, agent ID, score

agent_store = AgentStore(use_llm_agent_card=False)

def train_meta_model(train_data: Data, val_data: Data):
    """
    Train on data and evaluate on test questions.
    """
    model = FORCMetaModel(ModelConfig())
    trainer = FORCTrainer(model)
    num_batches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE
    epoch_losses = []

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(range(0, len(train_data), BATCH_SIZE), 
                    desc=f"  Epoch {epoch+1}/{NUM_EPOCHS} - Training",
                    total=num_batches)
        for i in pbar:
            batch = train_data[i:i+BATCH_SIZE]
            loss = trainer.train_step(batch)
            epoch_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = np.mean(epoch_losses)
        val_metrics = trainer.evaluate(val_data)
        print(f"    Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_metrics['f1']:.4f}")
        
    return model, trainer

def evaluate_meta_model(model: FORCMetaModel, trainer: FORCTrainer, test_questions: list[tuple[int, str]]):
    """
    Evaluate the model on test questions.
    """
    scores = []
    
    # Process questions one by one
    for (qid, question_text) in tqdm(test_questions, desc="Evaluating"):
        # Prepare all agent inputs for this question
        # question_inputs = [(question_text, agent_id) for agent_id in agents_df.index]
        
        matches = agent_store.match_by_qid(qid, top_k=8)
        inputs = [(qid, match.agent.id) for match in matches]

        # Process agents in batches for this question
        question_pred_probs = []
        for i in range(0, len(inputs), BATCH_SIZE):
            batch_inputs = inputs[i:i+BATCH_SIZE]
            
            # Process batch of agents for this question
            encoded = model.prepare_input(batch_inputs)
            input_ids = encoded['input_ids'].to(trainer.device)
            attention_mask = encoded['attention_mask'].to(trainer.device)
            predictions = model(input_ids, attention_mask)
            batch_pred_probs = predictions.cpu().detach().numpy().flatten()
            question_pred_probs.extend(batch_pred_probs)
        
        # Find best agent for this question
        agent_predictions = [(agent_id, pred_prob) for agent_id, pred_prob in zip([m.agent.id for m in matches], question_pred_probs)]
        best_agent_id, best_pred_prob = max(agent_predictions, key=lambda x: x[1])
        
        best_agent = agent_store.from_id(best_agent_id, shallow=True)
        actual_score = best_agent.has_sources(question_text)
        scores.append(actual_score)
    
    print(f"Result: {np.sum(scores)}/{len(scores)}")


if __name__ == "__main__":
    with open("data/test_qids.txt", "r") as f:
        test_qids = [int(qid.strip()) for qid in f.read().split(',')]
    
    test_questions = []
    for qid in test_qids:
        question_text = questions_df.loc[qid, 'question']
        test_questions.append((qid, question_text))
    
    for attack_volume in range(100, 101, 10):
        graph_data = pd.read_csv(f"data/graph_{attack_volume}.csv", 
                                 dtype={'source_agent': int, 'target_agent': int, 'question': int, 'score': float})
        data = []
        for _, row in graph_data.iterrows():
            question_text = questions_df.loc[row['question'], 'question']
            agent_id = row['target_agent']
            score = row['score']
            data.append((question_text, agent_id, score))

        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        model, trainer = train_meta_model(train_data, val_data)

        evaluate_meta_model(model, trainer, test_questions)
        break

        
        
    

