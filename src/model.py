import torch
import torch.nn as nn
class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.input_layer = nn.Linear(X.shape[1], 10)
    self.linear = nn.Linear(10, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x=self.input_layer(x)
    x=self.linear(x)
    x=self.sigmoid(x)
    return x
