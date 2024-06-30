import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, transaction_size, embedding_dim, hidden_dim, num_layers, device, num_classes = 1, batch_first = True, fc_hidden_dim = 128):
        super(LSTM, self).__init__()
        self.transaction_size = transaction_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.fc_hidden_dim = fc_hidden_dim
        
        # initialize embedding, LSTM and fully connected layers
        self.embedding_layer = nn.Embedding(self.transaction_size, self.embedding_dim, device=self.device)
        self.ln1 = nn.LayerNorm(self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, batch_first=self.batch_first, device=self.device)
        # -> in : (batch_size, sequence_length)
        # -> out: (batch_size, sequence_length, embedding_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, self.fc_hidden_dim, device=self.device), # Adjust the hidden layer size as needed
            nn.LayerNorm(self.fc_hidden_dim), 
            nn.ReLU(),  # Add ReLU activation
            nn.Dropout(p=0.5),
            nn.Linear(self.fc_hidden_dim, 1, device=self.device),  # Output layer
            nn.Sigmoid() # Add sigmoid activation
        )
        self.fc.to(self.device)
        
    def forward(self, x):
        
        x_embed = self.embedding_layer(x)
        x_embed = self.ln1(x_embed)
        
        h0 = torch.zeros(self.num_layers, x_embed.shape[0], self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x_embed.shape[0], self.hidden_dim).to(self.device)
        # print(h0.requires_grad)
        h_t, _ = self.lstm(x_embed.to(self.device), (h0, c0)) # (batch_size, sequence_length/ number of total hidden states, hidden_dim)
        h_t = self.ln2(h_t)
        out = self.fc(h_t[:, -1, :]) # (batch_size, 1)
        return out.to(self.device)
        
        