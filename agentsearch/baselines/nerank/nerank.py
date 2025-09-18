import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import random

from .hetero_embedding import MetapathWalkGenerator, HeterogeneousSkipGram
from .question_encoder import LSTMQuestionEncoder, QuestionTokenizer
from .cnn_scorer import CNNScoringFunction, RankingLoss


class NeRank(nn.Module):
    """
    NeRank: Network embedding-augmented Ranking for personalized question routing.
    
    Based on Li et al. (2019): "Personalized Question Routing via Heterogeneous Network Embedding"
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NeRank model.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Default configuration
        default_config = {
            'embedding_dim': 256,
            'lstm_hidden_dim': 256,
            'lstm_layers': 1,
            'cnn_num_filters': 64,
            'cnn_hidden_dim': 128,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_negative': 3,
            'metapath': 'AQRQA',
            'walk_length': 13,
            'walks_per_node': 20,
            'max_question_length': 100,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.config = {**default_config, **(config or {})}
        self.device = torch.device(self.config['device'])
        
        # Components will be initialized in setup()
        self.hetero_embedding = None
        self.question_encoder = None
        self.scoring_function = None
        self.tokenizer = None
        
        # Mappings
        self.node_to_id = {}
        self.id_to_node = {}
        self.node_types = {}
    
    def setup(self, graph_data: Dict[str, Any], question_texts: Dict[str, str]):
        """
        Setup the model with data.
        
        Args:
            graph_data: Dictionary containing network relationships
            question_texts: Dictionary mapping question IDs to text
        """
        # Build node mappings
        self._build_node_mappings(graph_data)
        
        # Initialize components
        num_nodes = len(self.node_to_id)
        
        # Heterogeneous network embedding
        self.hetero_embedding = HeterogeneousSkipGram(
            num_nodes=num_nodes,
            embedding_dim=self.config['embedding_dim'],
            num_negative=self.config['num_negative']
        ).to(self.device)
        
        # Question tokenizer and encoder
        self.tokenizer = QuestionTokenizer(max_length=self.config['max_question_length'])
        self.tokenizer.build_vocab(list(question_texts.values()))
        
        self.question_encoder = LSTMQuestionEncoder(
            vocab_size=len(self.tokenizer.vocab),
            embedding_dim=300,
            hidden_dim=self.config['lstm_hidden_dim'],
            num_layers=self.config['lstm_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # CNN scoring function
        self.scoring_function = CNNScoringFunction(
            embedding_dim=self.config['embedding_dim'],
            num_filters=self.config['cnn_num_filters'],
            hidden_dim=self.config['cnn_hidden_dim'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Store data
        self.graph_data = graph_data
        self.question_texts = question_texts
        
        # Move to device
        self.to(self.device)
    
    def _build_node_mappings(self, graph_data: Dict[str, Any]):
        """Build mappings between nodes and indices."""
        node_id = 0
        
        # Add raisers
        if 'raiser_questions' in graph_data:
            for raiser in graph_data['raiser_questions'].keys():
                if raiser not in self.node_to_id:
                    self.node_to_id[raiser] = node_id
                    self.id_to_node[node_id] = raiser
                    self.node_types[raiser] = 'R'
                    node_id += 1
        
        # Add questions
        all_questions = set()
        if 'raiser_questions' in graph_data:
            for questions in graph_data['raiser_questions'].values():
                all_questions.update(questions)
        if 'answerer_questions' in graph_data:
            for questions in graph_data['answerer_questions'].values():
                all_questions.update(questions)
        
        for question in all_questions:
            if question not in self.node_to_id:
                self.node_to_id[question] = node_id
                self.id_to_node[node_id] = question
                self.node_types[question] = 'Q'
                node_id += 1
        
        # Add answerers
        if 'answerer_questions' in graph_data:
            for answerer in graph_data['answerer_questions'].keys():
                if answerer not in self.node_to_id:
                    self.node_to_id[answerer] = node_id
                    self.id_to_node[node_id] = answerer
                    self.node_types[answerer] = 'A'
                    node_id += 1
    
    def train_embeddings(self, num_epochs: int = 10):
        """
        Train heterogeneous network embeddings using metapath walks.
        
        Args:
            num_epochs: Number of training epochs
        """
        # Generate walks
        walk_generator = MetapathWalkGenerator(self.graph_data, self.config['metapath'])
        walks = walk_generator.generate_walks(
            num_walks=self.config['walks_per_node'],
            walk_length=self.config['walk_length']
        )
        
        # Convert walks to training pairs
        training_pairs = []
        window_size = 4
        for walk in walks:
            for i, center_node in enumerate(walk):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i != j:
                        center_id = self.node_to_id.get(center_node)
                        context_id = self.node_to_id.get(walk[j])
                        if center_id is not None and context_id is not None:
                            training_pairs.append((center_id, context_id))
        
        # Train embeddings
        optimizer = optim.Adam(self.hetero_embedding.parameters(), lr=self.config['learning_rate'])
        
        for epoch in range(num_epochs):
            total_loss = 0
            random.shuffle(training_pairs)
            
            for i in tqdm(range(0, len(training_pairs), self.config['batch_size']), 
                         desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = training_pairs[i:i+self.config['batch_size']]
                
                center_ids = torch.tensor([p[0] for p in batch]).to(self.device)
                context_ids = torch.tensor([p[1] for p in batch]).to(self.device)
                
                # Generate negative samples
                neg_samples = []
                for _ in range(len(batch)):
                    negs = []
                    for _ in range(self.config['num_negative']):
                        neg_id = random.randint(0, len(self.node_to_id) - 1)
                        negs.append(neg_id)
                    neg_samples.append(negs)
                neg_samples = torch.tensor(neg_samples).to(self.device)
                
                # Forward pass
                loss = self.hetero_embedding(center_ids, context_ids, neg_samples)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(training_pairs):.4f}")
    
    def train_ranking(self, training_data: List[Dict], num_epochs: int = 5):
        """
        Train the ranking model.
        
        Args:
            training_data: List of training samples with raiser, question, answerers
            num_epochs: Number of training epochs
        """
        optimizer = optim.Adam(
            list(self.question_encoder.parameters()) + 
            list(self.scoring_function.parameters()),
            lr=self.config['learning_rate']
        )
        ranking_loss = RankingLoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for sample in tqdm(training_data, desc=f"Ranking Epoch {epoch+1}/{num_epochs}"):
                # Get embeddings
                raiser_id = self.node_to_id.get(sample['raiser'])
                if raiser_id is None:
                    continue
                    
                raiser_embed = self.hetero_embedding.get_embeddings(
                    torch.tensor([raiser_id]).to(self.device)
                )
                
                # Encode question
                question_text = self.question_texts.get(sample['question'], "")
                token_ids, length = self.tokenizer.tokenize(question_text)
                token_ids = torch.tensor([token_ids]).to(self.device)
                length = torch.tensor([length]).to(self.device)
                question_embed = self.question_encoder(token_ids, length)
                
                # Get answerer embeddings
                accepted_id = self.node_to_id.get(sample.get('accepted_answerer'))
                answered_ids = [self.node_to_id.get(a) for a in sample.get('other_answerers', [])]
                unanswered_ids = [self.node_to_id.get(a) for a in sample.get('unanswered', [])]
                
                # Filter out None values
                answered_ids = [aid for aid in answered_ids if aid is not None]
                unanswered_ids = [uid for uid in unanswered_ids if uid is not None]
                
                if accepted_id is None or not answered_ids or not unanswered_ids:
                    continue
                
                # Compute scores
                accepted_embed = self.hetero_embedding.get_embeddings(
                    torch.tensor([accepted_id]).to(self.device)
                )
                accepted_score = self.scoring_function(raiser_embed, question_embed, accepted_embed)
                
                # Sample one answered and one unanswered
                answered_id = random.choice(answered_ids)
                unanswered_id = random.choice(unanswered_ids)
                
                answered_embed = self.hetero_embedding.get_embeddings(
                    torch.tensor([answered_id]).to(self.device)
                )
                answered_score = self.scoring_function(raiser_embed, question_embed, answered_embed)
                
                unanswered_embed = self.hetero_embedding.get_embeddings(
                    torch.tensor([unanswered_id]).to(self.device)
                )
                unanswered_score = self.scoring_function(raiser_embed, question_embed, unanswered_embed)
                
                # Compute loss
                loss = ranking_loss(accepted_score, answered_score, unanswered_score)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Ranking Epoch {epoch+1}: Loss = {total_loss/len(training_data):.4f}")
    
    def rank_answerers(self, raiser: str, question: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidate answerers for a question from a specific raiser.
        
        Args:
            raiser: Raiser ID
            question: Question ID or text
            candidates: List of candidate answerer IDs
            
        Returns:
            List of (answerer_id, score) tuples sorted by score
        """
        self.eval()
        
        with torch.no_grad():
            # Get raiser embedding
            raiser_id = self.node_to_id.get(raiser)
            if raiser_id is None:
                # Return random ranking if raiser unknown
                return [(c, random.random()) for c in candidates]
            
            raiser_embed = self.hetero_embedding.get_embeddings(
                torch.tensor([raiser_id]).to(self.device)
            )
            
            # Encode question
            if question in self.question_texts:
                question_text = self.question_texts[question]
            else:
                question_text = question
            
            token_ids, length = self.tokenizer.tokenize(question_text)
            token_ids = torch.tensor([token_ids]).to(self.device)
            length = torch.tensor([length]).to(self.device)
            question_embed = self.question_encoder(token_ids, length)
            
            # Score each candidate
            scores = []
            for candidate in candidates:
                candidate_id = self.node_to_id.get(candidate)
                if candidate_id is None:
                    scores.append((candidate, 0.0))
                    continue
                
                candidate_embed = self.hetero_embedding.get_embeddings(
                    torch.tensor([candidate_id]).to(self.device)
                )
                score = self.scoring_function(raiser_embed, question_embed, candidate_embed)
                scores.append((candidate, score.item()))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            
        return scores