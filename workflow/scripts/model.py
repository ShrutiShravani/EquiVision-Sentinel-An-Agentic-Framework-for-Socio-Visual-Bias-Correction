import torch
import torch.nn as nn
import timm
from logger import logging

class ClinicalVit(nn.Module):
    def __init__(self, num_classes=14, metadata_dim=10):
        super(ClinicalVit, self).__init__()

        logging.info(f"Initializing ClinicalVit with {num_classes} classes and {metadata_dim} metadata features.")

        # 1. Pretrained ViT Backbone
        # num_classes=0 removes the original ImageNet head, giving us the 768-dim features
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

        # 2. Metadata Branch (MLP)
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU() # Added one more ReLU for better non-linearity
        )

        # 3. Fusion Head
        # ViT Base hidden size is 768. MLP output is 32. Total = 800
        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 512),
            nn.BatchNorm1d(512), # Added BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes) # No Sigmoid here (handled by BCEWithLogitsLoss)
        )

    def forward(self, image, metadata):
        # Vision features (Shape: [Batch, 768])
        v_feats = self.vit(image)
        
        # Metadata features (Shape: [Batch, 32])
        # Fix: correctly calling self.metadata_mlp
        m_feats = self.metadata_mlp(metadata)
        
        # Late Fusion: Concatenate along the feature dimension (dim=1)
        # Result Shape: [Batch, 800]
        combined = torch.cat((v_feats, m_feats), dim=1)
        
        # Final Classification Logits
        return self.classifier(combined)