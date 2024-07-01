import numpy as np
import csv
import torch.nn as nn
import torch
from models.RNN import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # RNN settings
# embedding_dim = 64
# hidden_dim = 64
# num_layers = 2

# ## parameter setting
# num_epochs = 50
# learning_rate = 0.001

# a = 11 * torch.ones([1])
# print(a)
# print(a.shape)

# model = RNN(312, embedding_dim, hidden_dim, num_layers, device, num_classes = 1, batch_first = True, fc_hidden_dim = 128)
# model(torch.zeros((100,10)), device)

m = nn.Conv1d(16, 35, 1, stride=1)
input = torch.randn(20, 16, 1)
output = m(input)
print(output.shape)
print(torch.squeeze(output).shape)