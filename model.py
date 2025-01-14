import torch.nn.init as init
import torch
import torch.nn as nn
import math

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_len, padding_idx, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_encoding = self._generate_sinusoidal_embeddings(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()  # ReLU activation function
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._initialize_weights)

    def _generate_sinusoidal_embeddings(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for Linear layers
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            if module.weight.dim() >= 2:  # Check if dimensions are sufficient
                init.xavier_uniform_(module.weight)
            else:
                init.normal_(module.weight, mean=0, std=0.1)  # Fallback for lower dimensions
        elif isinstance(module, nn.TransformerEncoderLayer):
            # Initialize Transformer weights
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    init.xavier_uniform_(param)
                elif "bias" in name:
                    init.zeros_(param)


    def forward(self, x):
        seq_len = x.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = self.embedding(x) + positional_encoding
        x = self.dropout(x)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer_encoder(x.transpose(0, 1), mask=mask)
        x = x.transpose(0, 1)
        x = self.fc_dropout(x)
        x = self.relu(x)  # Apply ReLU before final output layer
        x = self.fc_out(x)
        return x
