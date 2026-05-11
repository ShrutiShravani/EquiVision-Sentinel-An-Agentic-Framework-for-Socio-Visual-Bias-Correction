import torch.nn.functional as F
import torch
import torch.nn as nn


class WeightedFocalLoss(nn.Module):
    def __init__(self,alpha,gamma):
        super(WeightedFocalLoss,self).__init__()
        self.register_buffer('alpha', alpha)
        self.gamma=gamma

    def forward(self,inputs,targets,patient_weights):
        ## 1. Standard Binary Cross Entropy with Logits (Includes Sigmoid)
        bce_loss= F.binary_cross_entropy_with_logits(inputs,targets,reduction='none')

        # 2. Calculate pt (Probability of the correct class)
        pt= torch.exp(-bce_loss) #probability of correct class

        # 3. Apply Alpha (ETL Weights) and Gamma (Focusing)
        focal_loss= self.alpha *(1-pt)**self.gamma *bce_loss

        # 3. Apply Patient Weights (Soft-Capping)
        # patient_weights is [Batch], we unsqueeze to [Batch, 1] to multiply
        final_loss = focal_loss * patient_weights.unsqueeze(1)
        
        #4. Average the loss across the batch for the optimizer
        return final_loss.mean()
