import torch
import torch.nn as nn
from torchvision.models import resnet101



class ResNetEncoder(nn.Module):
    def __init__(self, config):
        super(ResNetEncoder, self).__init__()
        self.enc_hidden_dim = config.enc_hidden_dim
        self.dec_hidden_dim = config.dec_hidden_dim
        self.dec_num_layers = config.dec_num_layers
        self.pixel_size = self.enc_hidden_dim * self.enc_hidden_dim

        base_model = resnet101(pretrained=True, progress=False)
        base_model = list(base_model.children())[:-2]
        self.resnet = nn.Sequential(*base_model)  # output size: B x 2048 x H/32 x W/32
        self.pooling = nn.AdaptiveAvgPool2d((self.enc_hidden_dim, self.enc_hidden_dim))

        self.relu = nn.ReLU()
        self.hidden_dim_changer = nn.Sequential(
            nn.Linear(self.pixel_size, self.dec_num_layers),
            nn.ReLU()
        )
        self.h_mlp = nn.Linear(2048, self.dec_hidden_dim)
        self.c_mlp = nn.Linear(2048, self.dec_hidden_dim)

        self.fine_tune(True)


    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


    def forward(self, x):
        batch_size = x.size(0)

        x = self.resnet(x)
        x = self.pooling(x)
        x = x.view(batch_size, 2048, -1)

        if self.dec_num_layers != 1:
            tmp = self.hidden_dim_changer(self.relu(x))
        else:
            tmp = torch.mean(x, dim=2, keepdim=True)
        tmp = torch.permute(tmp, (2, 0, 1))
        h0 = self.h_mlp(tmp)
        c0 = self.c_mlp(tmp)
        return x, (h0, c0)