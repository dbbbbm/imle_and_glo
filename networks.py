import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 7 * 7 * 32),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 5, 2, 5//2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, 5, 2, 5//2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 1, 5, 1, 5//2, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 7, 7)
        x = self.conv(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feat, hid_feat),
            nn.BatchNorm1d(hid_feat),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_feat, out_feat),
        )
    
    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    g = Generator()
    z = torch.randn(16, 64)
    out = g(z)
    print(out.shape)
    