import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNScoringFunction(nn.Module):
    """CNN-based scoring function for ranking answerers."""
    
    def __init__(self, embedding_dim: int = 256, num_filters: int = 64, 
                 hidden_dim: int = 128, dropout: float = 0.1):
        """
        Initialize the CNN scoring function.
        
        Args:
            embedding_dim: Dimension of entity embeddings
            num_filters: Number of convolutional filters
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        
        # Three convolutional kernels with different widths
        # k1: captures features within each entity
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(embedding_dim, 1))
        
        # k2: captures pairwise correlations
        self.conv2 = nn.Conv2d(1, num_filters, kernel_size=(embedding_dim, 2))
        
        # k3: captures overall correlations across all three entities
        self.conv3 = nn.Conv2d(1, num_filters, kernel_size=(embedding_dim, 3))
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, raiser_embed: torch.Tensor, question_embed: torch.Tensor, 
                answerer_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking scores for answerers.
        
        Args:
            raiser_embed: Raiser embeddings (batch_size, embedding_dim)
            question_embed: Question embeddings (batch_size, embedding_dim)
            answerer_embed: Answerer embeddings (batch_size, embedding_dim)
            
        Returns:
            Ranking scores (batch_size,)
        """
        batch_size = raiser_embed.size(0)
        
        # Stack embeddings to create feature map
        # Shape: (batch_size, embedding_dim, 3)
        feature_map = torch.stack([raiser_embed, question_embed, answerer_embed], dim=2)
        
        # Add channel dimension for convolution
        # Shape: (batch_size, 1, embedding_dim, 3)
        feature_map = feature_map.unsqueeze(1)
        
        # Apply convolutional kernels
        # conv1: (batch_size, num_filters, 1, 3)
        h1 = self.relu(self.conv1(feature_map))
        h1 = h1.view(batch_size, self.num_filters, -1)
        h1 = F.max_pool1d(h1, kernel_size=h1.size(2))
        h1 = h1.squeeze(2)
        
        # conv2: (batch_size, num_filters, 1, 2)
        h2 = self.relu(self.conv2(feature_map))
        h2 = h2.view(batch_size, self.num_filters, -1)
        h2 = F.max_pool1d(h2, kernel_size=h2.size(2))
        h2 = h2.squeeze(2)
        
        # conv3: (batch_size, num_filters, 1, 1)
        h3 = self.relu(self.conv3(feature_map))
        h3 = h3.squeeze(-1).squeeze(-1)
        
        # Concatenate features
        combined = torch.cat([h1, h2, h3], dim=1)
        combined = self.dropout(combined)
        
        # Fully connected layers
        hidden = self.relu(self.fc1(combined))
        hidden = self.dropout(hidden)
        scores = self.fc2(hidden).squeeze(1)
        
        return scores


class RankingLoss(nn.Module):
    """Ranking loss for training the scoring function."""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize the ranking loss.
        
        Args:
            margin: Margin for ranking loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, accepted_scores: torch.Tensor, answered_scores: torch.Tensor,
                unanswered_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss.
        
        The loss ensures:
        1. Accepted answers score higher than other answers
        2. Answered questions score higher than unanswered ones
        
        Args:
            accepted_scores: Scores for accepted answers
            answered_scores: Scores for answered but not accepted
            unanswered_scores: Scores for unanswered questions
            
        Returns:
            Loss value
        """
        # Loss between accepted and answered
        loss1 = F.relu(self.margin - (accepted_scores - answered_scores))
        
        # Loss between answered and unanswered
        loss2 = F.relu(self.margin - (answered_scores - unanswered_scores))
        
        return loss1.mean() + loss2.mean()