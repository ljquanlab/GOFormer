import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_pos = loss_pos * ((1 - probs_pos) ** self.gamma_pos)
        probs_neg = (probs + self.clip).clamp(max=1)
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))
        loss_neg = loss_neg * (probs_neg ** self.gamma_neg)
        loss = -(loss_pos + loss_neg)
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce_loss
        return loss.mean()


class HierarchicalLoss(nn.Module):
    def __init__(self, go_graph, lambda_hier=0.1):
        super(HierarchicalLoss, self).__init__()
        self.lambda_hier = lambda_hier
        self.register_buffer('parent_matrix', go_graph)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_labels]
            targets: [batch_size, num_labels]
        """
        probs = torch.sigmoid(logits)
        parent_probs = torch.matmul(probs, self.parent_matrix)
        violation = F.relu(probs - parent_probs)
        hier_loss = violation.mean()
        return hier_loss


def get_loss_function(config, go_graph=None):

    if config.LOSS_TYPE == "asymmetric":
        base_loss = AsymmetricLoss(
            gamma_neg=config.ASL_GAMMA_NEG,
            gamma_pos=config.ASL_GAMMA_POS,
            clip=config.ASL_CLIP
        )
    elif config.LOSS_TYPE == "focal":
        base_loss = FocalLoss()
    else:
        base_loss = nn.BCEWithLogitsLoss()
    if go_graph is not None:
        hier_loss = HierarchicalLoss(go_graph)
        return lambda logits, targets: base_loss(logits, targets) + hier_loss(logits, targets)
    
    return base_loss