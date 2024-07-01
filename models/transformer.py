import torch, math
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
# from torch.nn.modules.transformer import TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerEncoder

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout = 0.0, max_len = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# A forcasting model
class TransformerClassifier(torch.nn.Module):
    def __init__(self, 
                 transaction_size = 312,
                 seq_len = 10,
                 embedding_dim = 32,
                 nhead = 2,
                 dim_feedforward_encoder = 128,
                 fc_hidden_dim = 128,
                 dropout = 0.0,
                 device = "cuda",
                 activation = "gelu",
                 batch_first = True,
                 norm_first = True,
                 bias = True,
                 num_layers = 6,
                 is_causal = True,
                 learn_pos_embed = True):
        super(TransformerClassifier, self).__init__()

        # Set Class-level Parameters
        self.transaction_size = transaction_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.d_model = self.embedding_dim
        self.nhead = nhead
        self.dim_feedforward_encoder = dim_feedforward_encoder
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout = dropout
        self.device = device
        self.activation = activation
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.bias = bias
        self.num_layers = num_layers
        self.is_causal = is_causal
        self.learn_pos_embed = learn_pos_embed

        # Input Embedding 
        # self.input_embedding  = nn.Linear(1, self.embedding_dim)
        self.input_embedding = nn.Embedding(self.transaction_size, self.embedding_dim, device=self.device)

        # Positional Encoder 
        if self.learn_pos_embed:
            self.position_embedding = nn.Embedding(self.seq_len, self.embedding_dim)
            
        else:
            self.position_embedding = PositionalEmbedding(d_model = self.d_model, 
                                                          dropout = self.dropout,
                                                          max_len = self.seq_len)
        
        # Transformer Encoder Layer
        self.transformer_encoder_layer = TransformerEncoderLayer(self.d_model,
                                                                 self.nhead,
                                                                 dim_feedforward = self.dim_feedforward_encoder,
                                                                 dropout = self.dropout,
                                                                 activation = self.activation,
                                                                 layer_norm_eps = 1e-05,
                                                                 batch_first = self.batch_first,
                                                                 norm_first = self.norm_first,
                                                                 bias = self.bias,
                                                                 device = self.device)
        
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer,
                                                      self.num_layers,
                                                      norm = None,
                                                      enable_nested_tensor = True,
                                                      mask_check = True)
        # 1x1 Conv to reduce dimension
        # self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        # Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(self.d_model * self.seq_len, self.fc_hidden_dim, device=self.device), # Adjust the hidden layer size as needed
            nn.ReLU(),  # Add ReLU activation
            nn.Dropout(p=0.5),
            # nn.Linear(self.fc_hidden_dim, 2 * self.fc_hidden_dim, device=self.device), # Adjust the hidden layer size as needed
            # nn.ReLU(),  # Add ReLU activation
            # nn.Dropout(p=0.5),
            # nn.Linear(2 * self.fc_hidden_dim, self.fc_hidden_dim, device=self.device), # Adjust the hidden layer size as needed
            # nn.ReLU(),  # Add ReLU activation
            nn.Linear(self.fc_hidden_dim, 1, device=self.device),  # Output layer
            nn.Sigmoid() # Add sigmoid activation
        )

    def forward(self, x):
        input_embed = self.input_embedding(x)
        pos_embed = self.position_embedding(torch.arange(x.shape[1], device=self.device))
        x = input_embed + pos_embed.unsqueeze(0).expand(900, -1, -1)
        
        if self.is_causal:
            # src_mask = self.create_lower_triangular_mask()
            src_mask = self.generate_square_subsequent_mask()
            src_mask.to(self.device)
        else:
            src_mask = None
        
        x = self.transformer_encoder(x.to(self.device), mask = src_mask, is_causal=self.is_causal)
        print('self.seq_len*self.d_model: ', self.seq_len * self.d_model)
        out = self.fc(x.reshape((-1, self.seq_len * self.d_model)))
        return out
    
    def create_lower_triangular_mask(self):
        # Create a lower-triangular mask with -inf values above the diagonal
        mask = torch.tril(torch.full((self.seq_len, self.seq_len), float('-inf')))
        return mask
    
    def generate_square_subsequent_mask(self):
        return torch.triu(torch.full((self.seq_len, self.seq_len), float('-inf'), dtype=torch.float32, device=self.device),diagonal=1)