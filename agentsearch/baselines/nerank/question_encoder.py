import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
import numpy as np


class LSTMQuestionEncoder(nn.Module):
    """LSTM encoder for question text representation."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 hidden_dim: int = 256, num_layers: int = 1, 
                 dropout: float = 0.1, pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Initialize the LSTM question encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pretrained_embeddings: Optional pretrained word embeddings
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode question text using LSTM.
        
        Args:
            input_ids: Token IDs of shape (batch_size, max_length)
            lengths: Actual lengths of sequences
            
        Returns:
            Question embeddings of shape (batch_size, hidden_dim)
        """
        batch_size = input_ids.size(0)
        
        # Get word embeddings
        embedded = self.word_embeddings(input_ids)
        embedded = self.dropout(embedded)
        
        # Pack sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Use final hidden state as question representation
        # Shape: (num_layers, batch_size, hidden_dim)
        if self.num_layers > 1:
            question_repr = hidden[-1]  # Take last layer
        else:
            question_repr = hidden.squeeze(0)
            
        return question_repr


class QuestionTokenizer:
    """Simple tokenizer for question text."""
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_length: int = 100):
        """
        Initialize the tokenizer.
        
        Args:
            vocab: Vocabulary mapping words to indices
            max_length: Maximum sequence length
        """
        self.vocab = vocab if vocab else {'<PAD>': 0, '<UNK>': 1}
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of question texts
            min_freq: Minimum frequency for a word to be included
        """
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add words meeting frequency threshold
        idx = len(self.vocab)
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.vocab:
                self.vocab[word] = idx
                idx += 1
    
    def tokenize(self, text: str) -> Tuple[List[int], int]:
        """
        Tokenize a single text.
        
        Args:
            text: Question text
            
        Returns:
            Token IDs and actual length
        """
        words = text.lower().split()[:self.max_length]
        token_ids = []
        
        for word in words:
            token_id = self.vocab.get(word, self.unk_token_id)
            token_ids.append(token_id)
        
        # Pad if necessary
        actual_length = len(token_ids)
        while len(token_ids) < self.max_length:
            token_ids.append(self.pad_token_id)
            
        return token_ids, actual_length
    
    def batch_tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of question texts
            
        Returns:
            Token IDs tensor and lengths tensor
        """
        batch_ids = []
        batch_lengths = []
        
        for text in texts:
            token_ids, length = self.tokenize(text)
            batch_ids.append(token_ids)
            batch_lengths.append(length)
        
        return torch.tensor(batch_ids), torch.tensor(batch_lengths)