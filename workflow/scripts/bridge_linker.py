from exception import HEPException
from logger import logging
import os,sys
import pandas as pd

def bridge_linkage_engine(clinical_df,adi_df):
    try:
        """
        Inputs:
        nih_clinical_df: Rows from NIH (Age, Gender, Label)
        wi_adi_df: Wisconsin ADI data by Pincode/Zip
        """
        master_records = []

        # Iterate through each NIH patient
        for _, patient in clinical_df.iterrows():
            age = patient['Patient Age']
            gender = patient['Patient Gender']
            
            # --- THE BRIDGE LOGIC (Based on Finding #3, #5, #6) ---
            
            # If High Risk Demographic (Elderly or Disconnected Youth)
            if age >= 65 or age <= 24:
                # Finding #3 & #6: Assign to high-deprivation/rural ADI blocks (Rank 8-10)
                sampled_context = adi_df[adi_df['ADI_STATERNK'] >= 8].sample(1)
            else:
                # Finding #5: General structural distribution
                sampled_context = adi_df.sample(1)

            # --- EXTRACTING THE INFORMATICS VECTOR ---
            adi_rank = sampled_context['ADI_STATERNK'].values[0]
            
            # Final Master Record
            record = {
                'image_index': patient['Image Index'],
                'pathology': patient['Finding Labels'],
                'age': age,
                'gender': gender,
                
                # Anchor Data from ADI Dataset
                'synthetic_zip': sampled_context['ZIP_4'].values[0],
                'adi_state_rank': adi_rank,
                
                # --- FEATURE ENGINEERING FROM FINDINGS ---
                
                # Finding #5: Binary flag for the "Top 15%" threshold
                'is_top_15_risk': 1 if adi_rank >= 8.5 else 0,
                
                # Finding #3: Rural Treatment Gap (1.25 Multiplier)
                'rural_penalty': 1.25 if (age >= 65 and adi_rank >= 9) else 1.0,
                
                # Finding #2: Pediatric Risk Multiplier
                'pediatric_idx': 1.20 if age <= 18 else 1.0,
                
                # Finding #3.1: Psychosocial Stress Factor (Gender Pay Gap Proxy)
                'psych_stress': 1.05 if gender == 'F' else 1.0
            }
            master_records.append(record)

        return pd.DataFrame(master_records)

    
    except Exception as e:
        raise HEPException(e,sys)

