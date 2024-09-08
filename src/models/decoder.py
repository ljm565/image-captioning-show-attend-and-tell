import torch
import torch.nn as nn
from models.modules import SoftAttention



class LSTMDecoder(nn.Module):
    def __init__(self, config, tokenizer):
        super(LSTMDecoder, self).__init__()
        self.pixel_size = config.enc_hidden_dim * config.enc_hidden_dim
        self.dec_hidden_dim = config.dec_hidden_dim
        self.dec_num_layers = config.dec_num_layers
        self.dropout = config.dropout
        self.using_attention = config.using_attention
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        if self.using_attention:
            self.attention = SoftAttention(self.dec_hidden_dim)
        self.input_size = self.dec_hidden_dim + 2048 if self.using_attention else self.dec_hidden_dim

        self.embedding = nn.Embedding(self.vocab_size, self.dec_hidden_dim, padding_idx=self.pad_token_id)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.dec_hidden_dim,
                            num_layers=self.dec_num_layers,
                            batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout) 
        self.relu = nn.ReLU()
        self.beta = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dec_hidden_dim, 2048),
            nn.Sigmoid()
        )     
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dec_hidden_dim, self.vocab_size)
        )

        self.embedding.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.bias.data.fill_(0)
            m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m, nn.Embedding):
            m.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x, hidden, enc_output):
        x = self.embedding(x)
        score = None

        gate = self.beta(hidden[0][-1])
        if self.using_attention:
            enc_output, score = self.attention(self.relu(enc_output), self.relu(hidden[0][-1]))
            enc_output = gate*enc_output
            x = torch.cat((x, enc_output.unsqueeze(1)), dim=-1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden, score

