import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNet(nn.Module):
    def __init__(self, args):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.gru = nn.GRU(32 * 4 * 4, 120, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(120 * 2, 84)
        self.fc2 = nn.Linear(84, args.n_obj)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x: [B, T, H, W]
        B, T, H, W = x.size()
        x = self.pool(F.relu(self.conv1(x.view(B * T//2, 2, H, W))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(B, T//2, 32 * 4 * 4).transpose(0, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = torch.cat([x[-1, :, :120], x[0, :, 120:]], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
