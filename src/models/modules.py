import torch
import torch.nn as nn
import torch.nn.functional as F



class SoftAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.hidden_size = hidden_size
        self.enc_wts = nn.Sequential(
            nn.Linear(2048, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.dec_wts = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.score_wts = nn.Linear(self.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def forward(self, enc_output, dec_hidden):
        enc_output = torch.permute(enc_output, (0, 2, 1))
        
        score = self.tanh(self.enc_wts(enc_output) + self.dec_wts(dec_hidden).unsqueeze(1))
        score = self.score_wts(score)
        score = F.softmax(score, dim=1)

        enc_output = torch.permute(enc_output, (0, 2, 1))
        enc_output = torch.bmm(enc_output, score).squeeze(-1)
        return enc_output, score