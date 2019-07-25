import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1,device=None):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(self.device)
        else:
            nll_weight = None
       
        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets): 
        
        targets = targets.squeeze(1) 
        loss = (1-self.jaccard_weight) * self.nll_loss(outputs,targets) 
        if self.jaccard_weight:
            eps = 1e-7
            for cls in range(self.num_classes): 
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp() 
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight 

        return loss
