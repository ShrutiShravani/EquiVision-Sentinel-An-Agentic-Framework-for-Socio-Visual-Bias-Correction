import torch
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from logger import logging  # Assuming your project logger
import os
from exception import HEPException

class CustomImageCSVDataset(Dataset):
    def __init__(self, df, global_norm_path):
        self.clinical_frame = df
        
        try:
            # FIX: Loading JSON values safely
            with open(global_norm_path, 'r') as f:
                norm_data = json.load(f)
                g_mean = norm_data['g_mean']
                g_std = norm_data['g_std']
            
            logging.info(f"Dataset initialized with {len(df)} samples.")
            logging.info(f"Applied Global Normalization -> Mean: {g_mean:.4f}, Std: {g_std:.4f}")
            
        except FileNotFoundError:
            logging.error(f"Normalization file not found at {global_norm_path}")
            raise
        except Exception as e:
            logging.error(f"Error initializing Dataset: {str(e)}")
            raise

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[g_mean]*3, std=[g_std]*3)
        ])
        
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
        self.clinical_cols = ['Patient Age', 'View Position_Encoded', 'ADI_National_Rank']

    def __len__(self):
        return len(self.clinical_frame)

    def __getitem__(self, idx):
        try:
            row = self.clinical_frame.iloc[idx]
            img_path = row['abs_image_path']
            
            # Diagnostic check for image existence
            if not os.path.exists(img_path):
                logging.warning(f"Image missing at index {idx}: {img_path}")
                # Optional: Return a different index or a blank tensor
            
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)

            # Metadata and Target extraction
            features = torch.tensor(row[self.clinical_cols].values.astype('float32'))
            target = torch.tensor(row[self.all_labels].values.astype('float32'))

            return image, features, target

        except Exception as e:
            # Log the specific image and patient ID that caused the failure
            patient_id = self.clinical_frame.iloc[idx].get('Patient ID', 'Unknown')
            logging.error(f"Error loading data at index {idx} (Patient: {patient_id}): {str(HEPException)}")
            
            # In production, we often return a neighboring sample to keep the batch size consistent
            return self.__getitem__(idx + 1 if idx + 1 < len(self) else 0)