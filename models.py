import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is expected to have shape [n_batches, input_size, d_model]
        x = x + self.pe
        return x

class CustomTransformerTimeSeries(nn.Module):
    def __init__(self, input_size, n_feat, hidden_size, num_layers, num_heads, dropout_prob=0.1):
        super(CustomTransformerTimeSeries, self).__init__()
        self.input_size = input_size
        self.n_feat = n_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        self.timeembed = nn.Linear(1, hidden_size)
        self.fc = nn.Linear(n_feat, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=input_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size),
            num_layers,
        )
        self.output = nn.Linear(hidden_size, n_feat)

    def forward(self, x, t):
        x = self.fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x + self.timeembed(t).view(-1,1,self.hidden_size))
        #x = x + self.timeembed(t).view(-1,1,self.hidden_size)
        x = self.output(x)
        return x