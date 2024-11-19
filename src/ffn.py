import torch
from torch import nn

import sys
from pathlib import Path

proj_path = Path('/cluster') / 'work' / 'jacobaal' / 'pers-pred'
proj_path = proj_path.resolve()
if proj_path not in sys.path: sys.path.append(str(proj_path))
from src.utils import get_commons

# TODO: UPDATE CONF
paths, constants, config, logger, device = get_commons(log=False, config_name='experiments4')

class Decoder(nn.Module):
    def __init__(self, hidden: list[int], dropout: float, final='none', norm=False):
        super(Decoder, self).__init__()
        layers = []
        for i in range(len(hidden) - 1):
            layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            if i < len(hidden) - 2:
                if norm: layers.append(nn.LayerNorm(hidden[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        if final.lower() == 'sigmoid': layers.append(nn.Sigmoid())
        elif final.lower() == 'relu': layers.append(nn.ReLU())
        elif final.lower() == 'none': pass
        else: raise TypeError('final activation function not recognized')
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, hidden, dropout, norm=False):
      super(Encoder, self).__init__()
      self.init_model(hidden, dropout, norm)

    def init_model(self, hidden, dropout, norm):
      layers = []
      for i in range(len(hidden) - 1):
          layers.append(nn.Linear(hidden[i], hidden[i + 1]))
          if i < len(hidden) - 2:
              if norm: layers.append(nn.LayerNorm(hidden[i + 1]))
              layers.append(nn.ReLU())
              layers.append(nn.Dropout(dropout))
      self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.model(x)

