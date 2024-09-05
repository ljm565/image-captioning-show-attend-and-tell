import torch
import torch.nn as nn
from models.modules import SoftAttention



class LSTMDecoder(nn.Module):
    def __init__(self, config, tokenizer):
        super(LSTMDecoder, self).__init__()
        self.pixel_size = config.enc_hidden_size * config.enc_hidden_size
        self.dec_hidden_size = config.dec_hidden_size
        self.dec_num_layers = config.dec_num_layers
        self.dropout = config.dropout
        self.is_attn = config.is_attn
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size
        if self.is_attn:
            self.attention = SoftAttention(self.dec_hidden_size)
        self.input_size = self.dec_hidden_size + 2048 if self.is_attn else self.dec_hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.dec_hidden_size, padding_idx=self.pad_token_id)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.dec_hidden_size,
                            num_layers=self.dec_num_layers,
                            batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout) 
        self.relu = nn.ReLU()
        self.beta = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.dec_hidden_size, 2048),
            nn.Sigmoid()
        )     
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dec_hidden_size, self.vocab_size)
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
        if self.is_attn:
            enc_output, score = self.attention(self.relu(enc_output), self.relu(hidden[0][-1]))
            enc_output = gate*enc_output
            x = torch.cat((x, enc_output.unsqueeze(1)), dim=-1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden, score

