import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from  policy_gradients.torch_utils import *

'''
Neural network models for estimating policy and value unctions
Contains:
- Initialization utilities
- Value Network(s)
- Policy Network(s)
- Retrieval Function
'''
HIDDEN_SIZES = (64, 64)
STD = 2**0.5

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    function borrowed from 
    '''
    #comment
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key") 


class PolicyNet(nn.Module):
    '''
    A continuous policy using a fully connected neural network.
    The parameterizing tensor is a mean and standard deviation vector, 
    which parameterize a gaussian distribution.
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes=None):
        super().__init__()
        self.activation_fx = nn.Tanh()
        self.action_dim = action_dim
        # self.discrete = False
        # self.time_in_state = time_in_state

        self.affine_layers = nn.ModuleList()

        if hidden_sizes!= None and isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes, hidden_sizes)
            # print(hidden_sizes)

        elif hidden_sizes == None:
            global HIDDEN_SIZES
            hidden_sizes = HIDDEN_SIZES

        prev_size = state_dim
        # hidden_sizes = [64, 64]
        # first layer state_dim (23) -> hidden_sizes[0] (64)
        # second layer hidden_sizes[0] (64) -> hidden_sizes[1] (64)
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i) 
            # initialize_weights(lin, init)
            self.affine_layers.append(lin)
            prev_size = i

        self.final_mean = nn.Linear(prev_size, action_dim)
        # final layer hidden_sizes[1] (64) -> action_dim (3)
        # initialize_weights(self.final_mean, init, scale=0.01)

        # # added to ignore weight-sharring
        # self.final_value = nn.Linear(prev_size, 1)
        self.final_mean.weight.data.mul_(0.1)
        self.final_mean.bias.data.mul_(0.0)
        # initialize_weights(self.final_value, init, scale=1.0)

        stdev_init = torch.zeros(1, action_dim)
        self.log_stdev = ch.nn.Parameter(stdev_init)

    def forward(self, x): 
        '''rets mean and std of the action distribution'''
        # If the time is in the state, discard it
        # if self.time_in_state:
        #     x = x[:,:-1]
        for affine in self.affine_layers:
            x = self.activation_fx(affine(x)) # activation for sure within -1, 1
        
        action_means = self.final_mean(x)
        log_stdev = self.log_stdev.expand_as(action_means)
        action_std = ch.exp(log_stdev)
        action_means = torch.clip(action_means, -1, 1) 
        return action_means, action_std, log_stdev

############################################
# Generic Value network, Value network MLP  #
############################################      
class ValueNet(nn.Module):
    '''
    An example value network, with support for arbitrarily many
    fully connected hidden layers (by default 2 * 128-neuron layers),
    maps a state of size (state_dim) -> a scalar value.
    '''
    def __init__(self, state_dim, hidden_sizes=None):
        '''
        Initializes the value network.
        Inputs:
        - state_dim, the input dimension of the network (i.e dimension of state)
        - hidden_sizes, an iterable of integers, each of which represents the size
        of a hidden layer in the neural network.
        Returns: Initialized Value network
        '''
        super().__init__()
        self.affine_layers = nn.ModuleList()
        self.activation_fx = nn.Tanh()

        if hidden_sizes is not None and isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes, hidden_sizes)
            # print(hidden_sizes)

        elif hidden_sizes == None:
            global HIDDEN_SIZES
            hidden_sizes = HIDDEN_SIZES

        prev = state_dim
        for h in hidden_sizes:
            l = nn.Linear(prev, h)
            # initialize_weights(l, init)
            self.affine_layers.append(l)
            prev = h

        self.final = nn.Linear(prev, 1)
        self.final.weight.data.mul_(0.1)
        self.final.bias.data.mul_(0.0)
        # initialize_weights(self.final, init, scale=1.0)

    def forward(self, x):
        '''
        Performs inference using the value network.
        Inputs:
        - x, the state passed in from the agent
        Returns:
        - The scalar (float) value of that state, as estimated by the net
        '''
        for affine in self.affine_layers:
            x = self.activation_fx(affine(x))
        state_value = self.final(x)
        return state_value

    def get_value(self, x):
        return self(x)

####################################################
#Discriminator network, Discriminator network MLP  #
####################################################  

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size, init):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # self.linear3.weight.data.mul_(0.1)
        # self.linear3.bias.data.mul_(0.0)
        initialize_weights(self.fc3, init, scale=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.fc3(x)
        return output


class Generator(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x



class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_dim, init):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)
        
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)
        initialize_weights(self.fc3, init, scale=1.0)

    def forward(self, x):
        x = self.d1(torch.tanh(self.fc1(x)))
        x = self.d2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x
    
class ClassifierWithAttention(nn.Module):
    def __init__(self, num_inputs, hidden_dim, init, attention_dim=10):
        super(ClassifierWithAttention, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Dropout layers
        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)
        
        # Attention mechanism
        self.attention_layer = nn.Linear(hidden_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1)

        # # Initialize weights for classification layer
        # self.fc3.weight.data.mul_(0.1)
        # self.fc3.bias.data.mul_(0.0)
        initialize_weights(self.fc3, init, scale=1.0)

    def forward(self, x):
        # First layer with dropout
        x = self.d1(torch.tanh(self.fc1(x)))
        
        # Attention mechanism
        attention_weights = torch.tanh(self.attention_layer(x))  # Compute attention scores
        attention_weights = torch.softmax(self.attention_score(attention_weights), dim=0)  # Normalize scores

        # Compute raw attention scores
        self.scores = torch.relu(self.attention_layer(x))
        self.scores = self.attention_score(self.scores)  # Shape: [batch_size, 1]
        self.scores = self.scores.squeeze(-1)  # Shape: [batch_size]

        # Normalize scores to obtain attention weights
        self.raw_attention_weights = F.softmax(self.scores, dim=0)  # Shape: [batch_size]
        
        # Apply attention weights
        x = x * attention_weights
        
        # Second layer with dropout
        x = self.d2(torch.tanh(self.fc2(x)))
        
        # Final classification layer
        x = self.fc3(x)
        return x
    
    def get_raw_attention_weights(self):

        return self.raw_attention_weights



# Need some refactors
class TransformerClassifier(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_heads = 2, num_layers =6, output_dim=1):
        super(TransformerClassifier, self).__init__()
        
        # Embedding layer to project inputs to hidden_dim
        self.embedding = nn.Linear(num_inputs, hidden_dim)
        
        # Positional encoding (optional if sequence structure is important)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_inputs, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1, 
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self.fc_out.weight.data.mul_(0.1)
        self.fc_out.bias.data.mul_(0.0)

    def forward(self, x):
        # Input embedding
        x = self.embedding(x) + self.positional_encoding
        
        # Transformer encoder
        x = x.permute(1, 0, 2)  # Transformer expects [sequence_length, batch_size, hidden_dim]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to [batch_size, sequence_length, hidden_dim]
        
        # Global average pooling across sequence length
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Fully connected output layer
        x = self.fc_out(x)
        return x
