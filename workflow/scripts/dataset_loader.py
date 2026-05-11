import torch
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from logger import logging  # Assuming your project logger
import os,sys
from exception import HEPException
import cv2

class CustomImageCSVDataset(Dataset):
    def __init__(self, df, global_norm_path,metadata_cols,target_size=(224,224)):
        self.clinical_frame = df
        self.clinical_cols=metadata_cols
        self.target_size= target_size
        
        try:
            # FIX: Loading JSON values safely
            with open(global_norm_path, 'r') as f:
                norm_data = json.load(f)
                g_mean = norm_data['global_mean']
                g_std = norm_data['global_std']
            
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
            '4146231', '315286', '254061', '4043731', '432551', 
            '4150821', '255848', '255573', '4292193', '435017', 
            '256441', '258780', '4053006', '201826'
        ]
        
    def process_image(self,img_path):
        #read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")

        # 2. Aspect Ratio Audit (Letterboxing/Padding)
        h, w = img.shape[:2]
        desired_size = self.target_size[0]
        ratio = float(desired_size) / max(h, w)
        new_size = tuple([int(x * ratio) for x in (w, h)])

        img = cv2.resize(img, (new_size[0], new_size[1]))

        # 3. Add Padding
        delta_w = desired_size - new_size[0]
        delta_h = desired_size - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        # 4. Convert to RGB (ViT requirement)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img_rgb # Return as uint8 numpy array for ToTensor()

    
    def __len__(self):
        return len(self.clinical_frame)

    def __getitem__(self, idx):
        try:
            row = self.clinical_frame.iloc[idx]
           
            img_np = self.process_image(row['abs_image_path'])
            if self.transform:
                image_tensor = self.transform(img_np)

            # Metadata and Target extraction
            features = torch.tensor(row[self.clinical_cols].values.astype('float32'))
            p_weight= torch.tensor(row['patient_weight'],dtype=torch.float32)
            target = torch.tensor(row[self.all_labels].values.astype('float32'))

            return image_tensor,p_weight, features, target

        except Exception as e:
            # Log the specific image and patient ID that caused the failure
            patient_id = self.clinical_frame.iloc[idx].get('Patient ID', 'Unknown')
            logging.error(f"Error loading data at index {idx} (Patient: {patient_id}): {str(e)}")
            raise HEPException(e,sys)