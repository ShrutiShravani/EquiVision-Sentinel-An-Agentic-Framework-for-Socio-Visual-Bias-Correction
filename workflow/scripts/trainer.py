import torch
from tqdm import tqdm
from logger import logging
import numpy as np
from sklearn.metrics import roc_auc_score


class ClinicalTrainer:
    def __init__(self, model, optimizer, criterion, device, accum_iter=4):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
        self.accum_iter = accum_iter
        
        logging.info(f"Trainer initialized on {device}")
        logging.info(f"Gradient Accumulation steps: {accum_iter}")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        # tqdm creates a beautiful progress bar in Cursor/Colab
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [Train]")
        
        for i, (images, metadata, labels) in pbar:
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = self.model(images, metadata)
                loss = self.criterion(outputs, labels) / self.accum_iter
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()

            # Weight update (Gradient Accumulation)
            if (i + 1) % self.accum_iter == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            current_batch_loss = loss.item() * self.accum_iter
            total_loss += current_batch_loss
            
            # Update progress bar every batch
            pbar.set_postfix({'loss': f"{current_batch_loss:.4f}"})

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch} Training Complete. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        logging.info(f"Starting Validation for Epoch {epoch}...")
        
        with torch.no_grad():
            for images, metadata, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(images, metadata)
                    loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Store for potential AUC calculation later
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        #concatenate all batches
        all_targets= np.vstack(all_targets)
        all_outputs= np.vstack(all_preds)
        
        auc_score= roc_auc_score(all_targets,all_outputs,average="macro")

        # Logging clinical metrics
        logging.info(f"Epoch {epoch} Validation Complete. Average Val Loss: {avg_val_loss:.4f}")
        
        # Optional: Log if the loss is exploding
        if np.isnan(avg_val_loss):
            logging.critical("CRITICAL: Validation Loss is NaN. Training likely failed.")
            
        return avg_val_loss