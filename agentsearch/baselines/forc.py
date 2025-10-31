import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from dataclasses import dataclass
from agentsearch.dataset.agents import agents_df, AgentStore, Agent
from agentsearch.dataset.questions import Question
from agentsearch.utils.globals import get_torch_device
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

NUM_EPOCHS = 5
BATCH_SIZE = 64
FORCData = tuple[str, int, int] # Question text, agent ID, binary_score

@dataclass
class ModelConfig:
    """Configuration for the FORC-style meta-model"""
    model_name: str = 'distilbert-base-uncased'  # DistilBERT model to use
    max_length: int = 512  # Maximum sequence length for tokenization
    dropout_rate: float = 0.1
    learning_rate: float = 0.00001 # 1e-5  # Learning rate used in FORC paper
    gradient_clipping: float = 0.1  # Gradient clipping value from FORC
    classification_threshold: float = 0.5  # Threshold for converting scores to binary labels


class FORCMetaModel(nn.Module):
    """
    Meta-model following the FORC paper architecture for binary classification.
    Uses DistilBERT (66M parameters) fine-tuned on query-agent pairs.
    
    The model takes text queries with appended agent tokens (e.g., "<LM_i>")
    and outputs a binary classification score predicting whether the agent 
    can successfully solve the query (trust score > threshold).
    """
    
    def __init__(self, config: ModelConfig):
        super(FORCMetaModel, self).__init__()
        self.config = config
        
        # Initialize DistilBERT model and tokenizer
        self.distilbert = DistilBertModel.from_pretrained(config.model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens for agents <LM_0>, <LM_1>, ..., <LM_n>
        # CRITICAL: Sort agent IDs to ensure deterministic token ordering
        agent_ids = sorted(agents_df.index.tolist())
        agent_tokens = [f"<LM_{id}>" for id in agent_ids]
        self.tokenizer.add_special_tokens({'additional_special_tokens': agent_tokens})
        
        # Resize token embeddings to accommodate new special tokens
        self.distilbert.resize_token_embeddings(len(self.tokenizer))
        
        # Binary classification head on top of DistilBERT
        # DistilBERT hidden size is 768
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(768, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FORC meta-model.
        
        Args:
            input_ids: Tensor of shape (batch_size, max_length)
                Tokenized input sequences with agent tokens appended
            attention_mask: Tensor of shape (batch_size, max_length)
                Attention mask for padding tokens
                
        Returns:
            scores: Tensor of shape (batch_size, 1)
                Predicted probability that the agent can solve the query
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch_size, 1)
        scores = self.sigmoid(logits)

        return scores
    
    def prepare_input(self, inputs: list[tuple[str, int]]) -> dict:
        """
        Prepare input for the model by tokenizing queries with agent tokens.
        
        Args:
            inputs: List of tuples containing (query text, agent_id)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        # Format: "<LM_iquery text"
        queries_with_agents = []
        for query, agent_id in inputs:
            agent_token = f"<LM_{agent_id}>"
            combined = f"{agent_token}{query}"
            queries_with_agents.append(combined)
        
        # Tokenize all queries with agent tokens
        encoded = self.tokenizer(
            queries_with_agents,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for binary classification.
        Using Binary Cross Entropy loss for binary labels.
        
        Args:
            predictions: Predicted probabilities from the model (after sigmoid)
            targets: Ground truth binary labels (0 or 1)
            
        Returns:
            loss: Scalar loss value
        """
        loss_fn = nn.BCELoss()  # Binary Cross Entropy for binary classification
        return loss_fn(predictions, targets)
    
    def configure_optimizer(self) -> optim.Optimizer:
        """
        Configure Adam optimizer with settings from FORC paper.
        
        Returns:
            optimizer: Configured Adam optimizer
        """
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)


class FORCTrainer:
    """
    Trainer class following FORC paper training procedure.
    - Adam optimizer with lr=3e-5
    - Gradient clipping at 0.1 (Euclidean norm)
    - Polynomial learning rate scheduler
    """
    
    def __init__(self, model: FORCMetaModel):
        self.device = get_torch_device()
        self.model = model.to(self.device)
        self.optimizer = model.configure_optimizer()
        self.gradient_clip_val = model.config.gradient_clipping
        
    def train_step(self, training_data: list[FORCData]) -> float:
        """
        Perform a single training step following FORC paper procedure.
        
        Args:
            training_data: List of tuples containing (query, agent_id, target_score)
                          where query is the text query, agent_id is the agent ID,
                          and target_score is the binary label (0.0 or 1.0)
            
        Returns:
            loss: Loss value for this step
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack the training data
        queries = [item[0] for item in training_data]
        agent_ids = [item[1] for item in training_data]
        targets = torch.tensor([item[2] for item in training_data], dtype=torch.float32).reshape(-1, 1)
        
        # Prepare input
        encoded = self.model.prepare_input(zip(queries, agent_ids))
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        targets = targets.to(self.device)
        
        predictions = self.model(input_ids, attention_mask)
        loss = self.model.compute_loss(predictions, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, evaluation_data: list[FORCData]) -> dict:
        """
        Evaluate the model on validation data.
        
        Args:
            evaluation_data: List of tuples containing (query, agent_id, target_score)
                            where query is the text query, agent_id is the agent ID,
                            and target_score is the binary label (0.0 or 1.0)
            
        Returns:
            metrics: Dictionary containing binary classification evaluation metrics
        """
        self.model.eval()
        
        # Unpack the evaluation data
        queries = [item[0] for item in evaluation_data]
        agent_ids = [item[1] for item in evaluation_data]
        targets = torch.tensor([item[2] for item in evaluation_data], dtype=torch.float32).reshape(-1, 1)
        
        with torch.no_grad():
            # Batch predictions with BATCH_SIZE
            all_predictions = []
            all_targets = []
            bce_losses = []
            for i in range(0, len(queries), BATCH_SIZE):
                batch_queries = queries[i:i+BATCH_SIZE]
                batch_agent_ids = agent_ids[i:i+BATCH_SIZE]
                batch_targets = targets[i:i+BATCH_SIZE]
                encoded = self.model.prepare_input(zip(batch_queries, batch_agent_ids))
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                batch_targets = batch_targets.to(self.device)
                predictions = self.model(input_ids, attention_mask)
                loss = self.model.compute_loss(predictions, batch_targets)
                bce_losses.append(loss.item() * len(batch_queries))
                all_predictions.append(predictions.detach().cpu())
                all_targets.append(batch_targets.detach().cpu())
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            bce_loss = sum(bce_losses) / len(queries)
            binary_preds = (predictions > 0.5).float()
            accuracy = torch.mean((binary_preds == targets).float()).item()
            
            # Precision, Recall, F1
            tp = torch.sum((binary_preds == 1) & (targets == 1)).float()
            fp = torch.sum((binary_preds == 1) & (targets == 0)).float()
            fn = torch.sum((binary_preds == 0) & (targets == 1)).float()
            tn = torch.sum((binary_preds == 0) & (targets == 0)).float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
        return {
            'loss': bce_loss,
            'accuracy': accuracy,
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'tp': tp.item(),
            'fp': fp.item(),
            'tn': tn.item(),
            'fn': fn.item()
        }
    
    def get_polynomial_decay_schedule(self, num_training_steps: int, 
                                     num_warmup_steps: int = 0,
                                     power: float = 1.0,
                                     last_epoch: int = -1):
        """
        Create polynomial decay learning rate scheduler as used in FORC paper.
        
        Args:
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            power: Power for polynomial decay
            last_epoch: The index of last epoch
            
        Returns:
            Learning rate scheduler
        """
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            ) ** power
        
        return LambdaLR(self.optimizer, lr_lambda, last_epoch)

def create_trained_meta_model(data: list[FORCData]) -> tuple[FORCMetaModel, FORCTrainer]:
    model = FORCMetaModel(ModelConfig())
    trainer = FORCTrainer(model)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
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

def forc_match(model: FORCMetaModel, trainer: FORCTrainer, agent_store: AgentStore, question: Question) -> list[Agent]:
    agents = [match.agent for match in agent_store.match(question, top_k=8)]
    inputs = [(question.text, agent.id) for agent in agents]

    # Process agents in batches for this question
    question_pred_probs = []
    for i in range(0, len(inputs), BATCH_SIZE):
        batch_inputs = inputs[i:i+BATCH_SIZE]
        encoded = model.prepare_input(batch_inputs)
        input_ids = encoded['input_ids'].to(trainer.device)
        attention_mask = encoded['attention_mask'].to(trainer.device)
        predictions = model(input_ids, attention_mask)
        batch_pred_probs = predictions.cpu().detach().numpy().flatten()
        question_pred_probs.extend(batch_pred_probs)
    
    # Find best agent for this question
    agent_predictions = [(agent_id, pred_prob) for agent_id, pred_prob in zip([agent.id for agent in agents], question_pred_probs)]
    agent_predictions.sort(key=lambda x: x[1], reverse=True)

    # Return top 8 agents based on prediction probabilities
    return list(map(lambda x: agent_store.from_id(x[0], shallow=True), agent_predictions[:8]))