import torch.nn as nn
import torch
import sys
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
        )

        self.values = nn.Sequential(
            nn.Linear(128, 1),
        )


        self.advantages = nn.Sequential(
            nn.Linear(128, 2)
        )


    def forward(self, inp):
        sys.stdout.flush()
        out = self.shared(torch.tensor(inp, dtype=torch.float32))

        advantages = self.advantages(out)
        values = self.values(out)

        advantages = F.softmax(advantages, dim=1)

        return advantages.float(), values.float()
