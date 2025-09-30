# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, X, y, ylabels, label, return_levels=True):
        self.X = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
        self.y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
        self.ylabels = ylabels if isinstance(ylabels, torch.Tensor) else torch.tensor(ylabels, dtype=torch.long)
        self.label = label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.bool)
        self.return_levels = return_levels

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.return_levels:
            return self.X[idx],self.y[idx],self.ylabels[idx],self.label[idx]
            
        else:
            return self.X[idx],self.label[idx]
            

class NLPHierarchyCorrector(nn.Module):
    def __init__(
            self, 
            input_dim,
            hidden_dim,
            n_classes_per_level
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for n_classes in n_classes_per_level
        ])

    def forward(self, x):
        output = self.encoder(x)
        return [head(output) for head in self.heads]
    
class NLPErrorPredictor(nn.Module):
    def __init__(
            self, 
            input_dim,
            hidden_dim
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        output = self.encoder(x)
        return [self.classifier(output)]