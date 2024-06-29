import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, transaction_size, num_layers, batch_size, use_gpu):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.embedding_layer = nn.Embedding(transaction_size, embedding_dim)
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # Adjust the hidden layer size as needed
            nn.ReLU(),  # Add ReLU activation
            nn.Linear(128, 1),  # Output layer
            nn.Sigmoid() # Add sigmoid activation
        )

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.embedding_layer(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.fc(lstm_out[-1])
        return y









import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),  # Adjust the hidden layer size as needed
            nn.ReLU(),  # Add ReLU activation
            nn.Linear(128, num_classes),  # Output layer
            nn.Sigmoid(dim=1)  # Add softmax activation
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step output
        return out