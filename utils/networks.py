"""The reconstruction network used by both stages of the ST-MRF quantification
pipeline.

Both stages — semi-solid MT (input: MT MRF frames + T1 + T2) and joint rNOE
(input: both rNOE MRF trajectories + T1, T2 and the stage-1 MT parameters) —
share the same architecture and only differ in input width, which each stage's
config supplies as `input_dim`: an input layer, `n_hidden` fully connected
hidden layers (ReLU -> dropout -> batch norm), and a sigmoid-bounded output
layer predicting the stage's two pool parameters.
"""

import torch.nn as nn


class Network(nn.Module):
    """input -> n_hidden x (ReLU, dropout, batch norm) -> sigmoid output."""

    def __init__(self, input_dim, n_hidden=2, n_neurons=300, dropout=0.2, output_dim=2):
        super().__init__()
        self.input = nn.Linear(input_dim, n_neurons)
        self.output = nn.Linear(n_neurons, output_dim)
        self.hidden = nn.ModuleList([nn.Linear(n_neurons, n_neurons) for _ in range(n_hidden)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_neurons) for _ in range(n_hidden)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input(x).relu()
        for layer, b_norm in zip(self.hidden, self.batch_norms):
            x = b_norm(self.dropout(layer(x).relu()))
        return self.output(x).sigmoid()
