"""
BERT Cross-Encoder Re-ranking Module

This module implements a cross-encoder based re-ranking system using BERT
to provide more accurate relevance scores for agent-question pairs.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

BATCH_SIZE = 64

@dataclass
class RerankResult:
    """Result from re-ranking with cross-encoder score"""
    agent_id: int
    original_rank: int
    original_score: float
    cross_encoder_score: float
    

class BERTCrossEncoderReranker:
    """
    Re-ranks agent matches using a BERT cross-encoder model that evaluates
    query-document pairs for more accurate relevance scoring.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Pre-trained cross-encoder model from HuggingFace.
                       Default uses MS MARCO trained model which is good for relevance.
                       Other options:
                       - "cross-encoder/ms-marco-MiniLM-L-12-v2" (more accurate, slower)
                       - "cross-encoder/qnli-distilroberta-base" (for Q&A tasks)
        """
        self.device = torch.device("mps")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def score_pairs(self, query: str, documents: list[str]) -> np.ndarray:
        """
        Score query-document pairs using the cross-encoder.
        
        Args:
            query: The question/query text
            documents: List of document texts (agent cards) to score against the query
            
        Returns:
            Array of relevance scores (0-1) for each document
        """
        scores = []
        
        # Process in batches for efficiency
        for i in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[i:i + BATCH_SIZE]
            
            # Create pairs of (query, document) for each document
            pairs = [[query, doc] for doc in batch_docs]
            
            # Tokenize pairs
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities using sigmoid (for binary classification)
                # Some models might need softmax instead
                batch_scores = torch.sigmoid(logits).cpu().numpy()
                
            scores.extend(batch_scores.flatten())
            
        return np.array(scores)
    
    def rerank(self, query: str, agent_matches: list, top_k: int = None, 
               show_progress: bool = False) -> list[RerankResult]:
        """
        Re-rank agent matches using cross-encoder scoring.
        
        Args:
            query: The question text
            agent_matches: List of AgentMatch objects from initial retrieval
            top_k: Return only top-k results after re-ranking (None = return all)
            show_progress: Show progress bar during re-ranking
            
        Returns:
            List of RerankResult objects sorted by cross-encoder score
        """
        if len(agent_matches) == 0:
            return []
            
        # Extract agent cards for scoring
        documents = [match.agent.agent_card for match in agent_matches]
        
        # Score all pairs
        if show_progress:
            print(f"Re-ranking {len(documents)} agents with cross-encoder...")
        
        scores = self.score_pairs(query, documents)
        
        # Create re-rank results
        results = []
        for i, (match, score) in enumerate(zip(agent_matches, scores)):
            results.append(RerankResult(
                agent_id=match.agent.id,
                original_rank=i,
                original_score=match.similarity_score,
                cross_encoder_score=float(score)
            ))
        
        # Sort by cross-encoder score (descending)
        results.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]
            
        return results
    
    def rerank_with_agents(self, query: str, agent_matches: list, top_k: int = None,
                           show_progress: bool = False) -> list:
        """
        Re-rank and return agent matches with updated scores.
        
        Args:
            query: The question text
            agent_matches: List of AgentMatch objects from initial retrieval
            top_k: Return only top-k results after re-ranking
            show_progress: Show progress bar during re-ranking
            
        Returns:
            List of AgentMatch objects with cross-encoder scores, sorted by relevance
        """
        rerank_results = self.rerank(query, agent_matches, top_k, show_progress)
        
        # Create mapping from agent_id to original AgentMatch
        id_to_match = {match.agent.id: match for match in agent_matches}
        
        # Return re-ordered AgentMatches with updated scores
        reranked_matches = []
        for result in rerank_results:
            original_match = id_to_match[result.agent_id]
            # Update the similarity score with cross-encoder score
            from agentsearch.dataset.agents import AgentMatch
            reranked_match = AgentMatch(
                agent=original_match.agent,
                similarity_score=result.cross_encoder_score
            )
            reranked_matches.append(reranked_match)
            
        return reranked_matches


class RerankingDataset(Dataset):
    """Dataset for fine-tuning the cross-encoder model"""
    
    def __init__(self, data: list[tuple[str, str, float]], tokenizer, max_length: int = 512):
        """
        Args:
            data: List of (question_text, agent_card, score) tuples
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        question_text, agent_card, score = self.data[idx]
        
        # Tokenize the question-agent pair
        encoding = self.tokenizer(
            question_text,
            agent_card,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float)
        }


def fine_tune_reranker(
    train_data: list[tuple[str, str, float]], 
    val_data: list[tuple[str, str, float]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    device: str = 'mps'
) -> BERTCrossEncoderReranker:
    """
    Fine-tune a BERT cross-encoder model for re-ranking.
    
    Based on the approach described in Bhopale & Tiwari (2023):
    "Transformer based contextual text representation framework for intelligent information retrieval"
    
    Args:
        train_data: Training data as list of (question_text, agent_card, score) tuples
        val_data: Validation data in same format
        model_name: Pre-trained model to start from
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        warmup_steps: Number of warmup steps for scheduler
        device: Device to train on (None for auto-detect)
        
    Returns:
        Fine-tuned BERTCrossEncoderReranker model
    """
    
    device = torch.device(device)
    print(f"Fine-tuning on device: {device}")
    
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = RerankingDataset(train_data, tokenizer)
    val_dataset = RerankingDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    model.train()
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training phase
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits.squeeze()
            
            # Calculate loss (MSE for regression or BCE for binary classification)
            # Using MSE as the paper mentions scoring relevance
            loss = F.mse_loss(torch.sigmoid(logits), labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits.squeeze()
                loss = F.mse_loss(torch.sigmoid(logits), labels)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps

        print(f"Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")
        
        model.train()
    
    # Create and return the fine-tuned reranker
    model.eval()
    reranker = BERTCrossEncoderReranker(model_name=model_name)
    reranker.model = model
    reranker.tokenizer = tokenizer
    reranker.device = device
    
    return reranker