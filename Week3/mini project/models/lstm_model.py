"""
PyTorch Multi-layer LSTM Architecture for Character-level Text and Music Generation.
Features:
- Embedding -> Multi-layer LSTM with Dropout -> Fully Connected Projection
- Temperature-scaled stochastic sampling
- Top-K sampling filtering
- Text & Music streaming/batch generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(TextLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout

        # Embedding layer
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Decoder projection to vocab logits
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: tensor of shape (batch_size, seq_len)
        hidden: tuple (h_0, c_0)
        """
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # (batch_size, seq_len, embedding_dim)
        embeds = self.encoder(x)
        embeds = self.dropout(embeds)

        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)

        # logits: (batch_size, seq_len, vocab_size)
        logits = self.decoder(lstm_out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        h_0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = weight.new_zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)


def sample_next_token(logits, temperature=1.0, top_k=None):
    """
    Applies temperature scaling and optional top-k filtering,
    then samples from the categorical distribution.
    
    Args:
        logits: 1D PyTorch tensor of unnormalized log-probabilities.
        temperature: float > 0. Lower = more predictable, Higher = more creative/diverse.
        top_k: int > 0. If specified, only sample among the top-k most probable tokens.
    """
    if temperature <= 0.01:
        # Argmax / greedy sampling
        return torch.argmax(logits).item()

    # Scale logits by temperature
    scaled_logits = logits / max(temperature, 1e-4)

    if top_k is not None and top_k > 0:
        k = min(top_k, scaled_logits.size(-1))
        values, indices = torch.topk(scaled_logits, k)
        probs = F.softmax(values, dim=-1)
        chosen_k_idx = torch.multinomial(probs, num_samples=1).item()
        return indices[chosen_k_idx].item()
    else:
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


def generate_sequence(
    model,
    char2idx,
    idx2char,
    seed_text,
    max_length=200,
    temperature=0.8,
    top_k=40,
    device=torch.device('cpu')
):
    """
    Generates novel text or ABC music notation using seed text priming.
    
    Args:
        model: Trained TextLSTM model
        char2idx: Dict mapping character to index
        idx2char: Dict mapping index to character
        seed_text: Initial string prompt
        max_length: Maximum number of characters to generate
        temperature: Creativity slider (0.1 - 2.0)
        top_k: Top-k sampling constraint (None or int)
        device: CPU or CUDA device
    """
    model.eval()
    
    if not seed_text:
        seed_text = " "

    # Filter seed_text to characters known in vocabulary
    valid_seed = [c for c in seed_text if c in char2idx]
    if not valid_seed:
        # Default to the first character in vocab
        valid_seed = [list(char2idx.keys())[0]]

    # Warm-up / prime hidden states with the seed text
    hidden = model.init_hidden(1, device)
    input_indices = [char2idx[c] for c in valid_seed]
    input_tensor = torch.tensor([input_indices], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, hidden = model(input_tensor, hidden)
        last_logit = logits[0, -1, :]

    generated_chars = list(valid_seed)

    # Autoregressive generation loop
    with torch.no_grad():
        for _ in range(max_length):
            next_idx = sample_next_token(last_logit, temperature=temperature, top_k=top_k)
            next_char = idx2char.get(next_idx, ' ')
            generated_chars.append(next_char)

            # Next step input
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long, device=device)
            logits, hidden = model(input_tensor, hidden)
            last_logit = logits[0, 0, :]

    return "".join(generated_chars)
