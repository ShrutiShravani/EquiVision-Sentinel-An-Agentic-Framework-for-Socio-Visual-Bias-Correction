from workflow.scripts.trainer import ClinicalTrainer
from workflow.scripts.dataset_loader import CustomImageCSVDataset
from workflow.scripts.weighted_focal_loss import WeightedFocalLoss
from workflow.scripts.model import ClinicalVit
from sklearn.model_selection import GroupKFold
from logger import logging
from exception import HEPException
from torch.utils.data import Dataset, DataLoader
from utils.load_config import load_config
import json
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import os,sys
import mlflow


path= load_config(config_path="config\path.yaml")


def run_pipeline(study_mode):
    try:

        omop_mapping = {
        '4146231': 'Atelectasis', '315286': 'Cardiomegaly', '254061': 'Consolidation',
        '4043731': 'Edema', '432551': 'Effusion', '4150821': 'Emphysema',
        '255848': 'Fibrosis', '255573': 'Hernia', '4292193': 'Infiltration',
        '435017': 'Mass', '256441': 'Nodule', '258780': 'Pleural_Thickening',
        '4053006': 'Pneumonia', '201826': 'Pneumothorax'
    }
        target_cols = list(omop_mapping.keys())

        metadata_configs = {
        "M1": ['Patient Age', 'gender_concept_id', 'View Position'],
        "M2": ['Patient Age', 'gender_concept_id', 'View Position', 'ADI_STATERNK'],
        "M3": ['Patient Age', 'gender_concept_id', 'View Position', 'ADI_STATERNK', 
               'is_top_15_risk', 'rural_penalty', 'pediatric_idx', 'psych_stress']
    }
        selected_meta = metadata_configs[study_mode]
        logging.info(f"Model {study_mode} is training on features: {selected_meta}")
        meta_dim= len(selected_meta)

        logging.info(f"!!! STARTING ABLATION STUDY: {study_mode} !!!")
        logging.info(f"Features used: {selected_meta}")

        all_labels= [
            '4146231', '315286', '254061', '4043731', '432551', 
            '4150821', '255848', '255573', '4292193', '435017', 
            '256441', '258780', '4053006', '201826'
        ]

        # 1. Device Handshake
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Targeting Device: {device}")

        # 2. Load Master Data
        master_df_path = path['MASTER_PATH'] # Update to your actual path
        if not os.path.exists(master_df_path):
            logging.error(f"Master CSV not found at {master_df_path}")
            return
            
        master_df = pd.read_csv(master_df_path)
        logging.info(f"Successfully loaded Master CSV with {len(master_df)} rows.")
        

        #set ML Flow experiment
        mlflow.set_experiment("Clinical-ViT-Socio-Clinical-Study")

        with mlflow.start_run(run_name=f"Study_{study_mode}_KFold_Training_Session"):
            #log global hyperparameters
            mlflow.log_params({
                "model_type": "vit_base_patch16_224",
                "learning_rate": 1e-4,
                "batch_size": 32,
                "epochs_per_fold": 10,
                "optimizer": "AdamW"
            })
            mlflow.log_param("metadata_features",selected_meta)
            mlflow.log_param("metadata_dim",meta_dim)

        # 3. STEP 1: HOLDOUT TEST SPLIT (10% of Patients)
        patient_ids = master_df['Patient ID'].unique()
        train_val_ids, test_ids = train_test_split(patient_ids, test_size=0.1, random_state=42)
        
        dev_df = master_df[master_df['Patient ID'].isin(train_val_ids)].reset_index(drop=True)
        test_df = master_df[master_df['Patient ID'].isin(test_ids)].reset_index(drop=True)
        
        os.makedirs("data/processed", exist_ok=True)
        test_df.to_csv("data/processed/test_holdout.csv", index=False)
        logging.info(f"Saved Test Holdout (Patients: {len(test_ids)})")

        # 4. K-FOLD ON DEV DATA
        gkf = GroupKFold(n_splits=5)
        os.makedirs("models/checkpoints", exist_ok=True)
        fold_results = {}

        for fold, (train_idx, val_idx) in enumerate(gkf.split(dev_df, groups=dev_df['Patient ID'])):
            logging.info(f"\n" + "="*30 + f"\n STARTING FOLD {fold+1}/5 \n" + "="*30)
            
            with mlflow.start_run(run_name=f"{study_mode}_Fold_{fold+1}", nested=True):
                # Create fold-specific dataframes
                train_sub_df = dev_df.iloc[train_idx].reset_index(drop=True)
                val_sub_df = dev_df.iloc[val_idx].reset_index(drop=True)

                # Initialize Datasets
                train_ds = CustomImageCSVDataset(train_sub_df,"data/processed/glob_norm.json",metadata_cols=selected_meta)
                val_ds = CustomImageCSVDataset(val_sub_df, "data/processed/glob_norm.json",metadata_cols=selected_meta)

                train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

                # --- MODEL INITIALIZATION ---
                # This is where your ClinicalVit class is called!
                model = ClinicalVit(num_classes=14, metadata_dim=meta_dim)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                
                # --- WEIGHTED FOCAL LOSS ---
                with open("data/processed/weights.json", "r") as f:
                    weights_dict = json.load(f)
                alpha_values= [weights_dict[omop_mapping[cid]] for cid in target_cols]
                alpha = torch.tensor(alpha_values).float().to(device)
                criterion = WeightedFocalLoss(alpha=alpha, gamma=2)

                # --- TRAINER ---
                trainer = ClinicalTrainer(model, optimizer, criterion, device=device, accum_iter=4)
        
                best_val_auc=0.0
                
                # 5. Training Loop
                for epoch in range(1, 11): 
                    train_loss = trainer.train_epoch(train_loader, epoch)
                    val_loss,val_auc = trainer.validate(val_loader, epoch,fold=fold + 1,class_names=all_labels)
                    
                    #log metrics to mlflow
                    mlflow.log_metric(f"train_loss_fold_{fold+1}",train_loss,step=epoch)
                    
                    
                    logging.info(f"Fold {fold+1} | Epoch {epoch} | Val Loss: {val_loss:.4f} | Val AUC :{val_auc:.4f}")

                    if val_auc >best_val_auc:
                        best_val_auc = val_auc
                        checkpoint_path = f"models/checkpoints/best_model_fold{fold+1}.pth"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'val_auc': val_auc,
                            'loss':val_loss,
                        }, checkpoint_path)


                        #log best model as an artifact for this fold
                        mlflow.log_artifact(checkpoint_path)
                 
                fold_results[fold+1] = {"best_auc":best_val_auc}
                mlflow.log_metric(f"final_best_auc_fold_{fold+1}",best_val_auc)
                logging.info(f"Training Complete. CV Results: {fold_results}")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise HEPException (e,sys)


if __name__=="__main__":
    run_pipeline(study_mode='M3') 