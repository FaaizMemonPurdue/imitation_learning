import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class CULoss(nn.Module):
    def __init__(self, conf, beta, non=False):
        super(CULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non
        if conf.mean() > 0.5:
            self.UP = True
        else:
            self.UP = False

    def forward(self, conf, labeled, unlabeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))
        
        if self.UP:
            #conf_risk = torch.mean((1-conf) * (y_conf_neg - y_conf_pos) + (1 - self.beta) * y_conf_pos)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, torch.ones(unlabeled.shape).to(device)))
            neg_risk = torch.mean((1 - conf) * y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * y_conf_pos) + unlabeled_risk
        else:
            #conf_risk = torch.mean(conf * (y_conf_pos - y_conf_neg) + (1 - self.beta) * y_conf_neg)
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, -torch.ones(unlabeled.shape).to(device)))
            pos_risk = torch.mean(conf * y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * y_conf_neg) + unlabeled_risk
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk
        return objective


class PNLoss(nn.Module):
    def __init__(self):
        super(PNLoss, self).__init__()
        self.loss = nn.SoftMarginLoss()

    def forward(self, conf, labeled):
        y_conf_pos = self.loss(labeled, torch.ones(labeled.shape).to(device))
        y_conf_neg = self.loss(labeled, -torch.ones(labeled.shape).to(device))

        objective = torch.mean(conf * y_conf_pos + (1 - conf) * y_conf_neg)
        return objective




class ConfULoss(nn.Module):
    """
    Class for Confident-Unlabeled Loss.
    
    Parameters:
        beta (float): Balancing factor for unlabeled data loss.
        non (bool): If True, applies clamping to risk terms.
    """
    def __init__(self, beta, non=False):
        super(ConfULoss, self).__init__()
        self.loss = nn.SoftMarginLoss()
        self.beta = beta
        self.non = non

    def forward(self, conf, labeled, unlabeled):
        """
        Compute the loss based on labeled and unlabeled data.

        Args:
            conf (Tensor): Confidence scores for the unlabeled data.
            labeled (Tensor): Labeled data tensor.
            unlabeled (Tensor): Unlabeled data tensor.

        Returns:
            Tensor: Computed loss.
        """
        device = labeled.device
        labeled_ones = torch.ones_like(labeled)
        labeled_neg_ones = -labeled_ones

        unlabeled_ones = torch.ones_like(unlabeled)
        unlabeled_neg_ones = -unlabeled_ones

        # Compute risks for labeled data
        y_conf_pos = self.loss(labeled, labeled_ones.to(device))
        y_conf_neg = self.loss(labeled, labeled_neg_ones.to(device))

        # Determine if majority confidence is positive
        UP = conf.mean() > 0.5


        if UP:
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, unlabeled_ones.to(device)))
            neg_risk = torch.mean((1 - conf) * y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * y_conf_pos) + unlabeled_risk
        else:
            unlabeled_risk = torch.mean(self.beta * self.loss(unlabeled, unlabeled_neg_ones.to(device)))
            pos_risk = torch.mean(conf * y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * y_conf_neg) + unlabeled_risk

        # Apply clamping if self.non is True
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk

        return objective



class AttentionConfULoss(nn.Module):
    def __init__(self, beta, non=False):
        """
        Confident-Unlabeled Loss with Attention Mechanism.
        
        Args:
            beta (float): Balancing factor for unlabeled data loss.
            non (bool): If True, applies clamping to risk terms.
        """
        super(AttentionConfULoss, self).__init__()
        self.loss = nn.SoftMarginLoss(reduction='none')  # No reduction to apply attention weights
        self.beta = beta
        self.non = non

    def forward(self, conf, labeled, unlabeled, attention_weights):
        """
        Compute the loss based on labeled and unlabeled data and attention weights.

        Args:
            conf (Tensor): Confidence scores for the unlabeled data (shape: [batch_size]).
            labeled (Tensor): Labeled data tensor (shape: [batch_size, num_classes]).
            unlabeled (Tensor): Unlabeled data tensor (shape: [batch_size, num_classes]).
            attention_weights (Tensor): Attention scores for each data point (shape: [batch_size]).
        
        Returns:
            Tensor: Computed loss.
        """
        device = labeled.device
        labeled_ones = torch.ones_like(labeled)
        label_neg_ones = -labeled_ones
        unlabel_ones = torch.ones_like(unlabeled)
        unlabel_neg_ones = -unlabel_ones

        # Compute risks for labeled data
        y_conf_pos = self.loss(labeled, labeled_ones)
        y_conf_neg = self.loss(labeled, label_neg_ones)

        # Compute risks for unlabeled data
        unlabeled_pos_risk = self.loss(unlabeled, unlabel_ones)
        unlabeled_neg_risk = self.loss(unlabeled, unlabel_neg_ones)

        # Apply attention weights to risks
        attention_weights = attention_weights.to(device)
        weighted_y_conf_pos = torch.mean(attention_weights * y_conf_pos, dim=-1)
        weighted_y_conf_neg = torch.mean(attention_weights * y_conf_neg, dim=-1)

        if conf.mean() > 0.5:
            unlabeled_risk = torch.mean(attention_weights * self.beta * unlabeled_pos_risk, dim=-1)
            neg_risk = torch.mean((1 - conf) * weighted_y_conf_neg)
            pos_risk = torch.mean((conf - self.beta) * weighted_y_conf_pos) + unlabeled_risk
        else:
            unlabeled_risk = torch.mean(attention_weights * self.beta * unlabeled_neg_risk, dim=-1)
            pos_risk = torch.mean(conf * weighted_y_conf_pos)
            neg_risk = torch.mean((1 - self.beta - conf) * weighted_y_conf_neg) + unlabeled_risk

        # Apply clamping if self.non is True
        if self.non:
            objective = torch.clamp(neg_risk, min=0) + torch.clamp(pos_risk, min=0)
        else:
            objective = neg_risk + pos_risk

        # Optional regularization: Encourage diverse attention weights
        reg_loss = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
        objective += 0.01 * reg_loss  # Add small regularization term

        return objective


# class AttentionWeightCalculator(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         """
#         Initialize the AttentionWeightCalculator.
#         Args:
#             input_dim (int): Dimension of the input features.
#             hidden_dim (int): Dimension of the hidden layer.
#         """
#         super(AttentionWeightCalculator, self).__init__()
#         self.attention_fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)  # Output a single score per data point
#         )
    
#     def forward(self, inputs):
#         """
#         Compute attention weights for each data point.
#         Args:
#             inputs (Tensor): Input feature tensor of shape [batch_size, input_dim].
#         Returns:
#             Tensor: Attention weights of shape [batch_size].
#         """
#         # Compute raw attention scores
#         scores = self.attention_fc(inputs)  # Shape: [batch_size, 1]
#         scores = scores.squeeze(-1)  # Shape: [batch_size]

#         # Normalize scores to obtain attention weights
#         attention_weights = F.softmax(scores, dim=0)  # Shape: [batch_size]
#         return attention_weights

# # Example Usage
# batch_size, input_dim, hidden_dim = 8, 10, 32
# inputs = torch.randn(batch_size, input_dim)

# # Initialize attention weight calculator
# attention_calculator = AttentionWeightCalculator(input_dim, hidden_dim)

# # Compute attention weights
# attention_weights = attention_calculator(inputs)

# print("Attention Weights:", attention_weights)
