import torch.nn as nn


class EncoderF(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)  # z


class ProjectorP(nn.Module):
    def __init__(self, embed_dim=256, input_dim=100):
        super().__init__()
        self.mask_embedding = nn.Linear(input_dim, embed_dim)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, z, mask):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        m_embed = self.mask_embedding(mask)
        h = z + m_embed
        return self.out(h)