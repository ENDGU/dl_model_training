import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=[256,128], dropout=0.2):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, num_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
