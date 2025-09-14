import torch
  import torch.nn as nn

  class MLP(nn.Module):
      def __init__(
          self,
          in_features,
          out_features,
          hidden_features=None,
          num_layers=2,
          dropout=0.0,
          activation=nn.ReLU,
      ):
          super().__init__()

          hidden_features = hidden_features or 2 * out_features

          layers = []
          prev_dim = in_features

          for i in range(num_layers - 1):
              layers.append(nn.Linear(prev_dim, hidden_features))
              layers.append(activation())
              if dropout > 0:
                  layers.append(nn.Dropout(dropout))
              prev_dim = hidden_features

          layers.append(nn.Linear(prev_dim, out_features))

          self.mlp = nn.Sequential(*layers)

      def forward(self, x):
          return self.mlp(x)