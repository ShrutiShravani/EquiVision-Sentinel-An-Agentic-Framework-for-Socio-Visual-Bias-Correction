import numpy as np
import pandas as pd
from exception import HEPException
from logger import logging
import os,sys
import json
from utils.load_config import load_config

path= load_config(config_path="config\path.yaml")

def run_clinical_etl(input_path,output_path):
    try:
        logging.info("---Starting clinical etl pipeline---")
        df =pd.read_csv(input_path)
        initial_count = len(df)

        #handle age outliers by dropping pateint above 100
    
        df['Patient Age'] = df['Patient Age'].clip(0, 100)
        logging.info(f"patient age clipped")


    #visual audit to chcek if images match wiht imageslink in clinical
    
    
        image_location_map={}

        #recrusive scan through all fodlers
        for root,dirs,files in os.walk("data\images"):
            for file in files:
                if file.endswith(".png"):
                    image_location_map[file]= os.path.abspath(os.path.join(root,file))


        found_images=set(image_location_map.keys())
        df =df[df['Image Index'].isin(found_images)].copy()
        if df.empty:
            raise HEPException("No matching images found in 'Image Index' vs data/images.", sys)
        df.reset_index(drop=True, inplace=True)


        #map image index to full image location
        df['abs_image_path']= df['Image Index'].map(image_location_map)
        
        logging.info(f"length of df {len(df)}")
    
    #multi -label  binarization:perform one hot encoding

        all_labels= [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
            'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
        
        

            #calculate weights for the loss function to be used for performance engineering

        total_samples= len(df)
        class_weights={}
        
        for label in all_labels:
            logging.info(label)

            count= df['Finding Labels'].apply(lambda x:label in x ).sum()

            # Inverse Frequency formula: (Total / Count) 
        # This gives rare classes like Hernia a very high weight
        # and common classes like Infiltration a low weight.
            weight= total_samples/(count+ 1e-6)
            class_weights[label] = round(weight,4)
        
        os.makedirs(os.path.dirname(path['WEIGHTS_PATH']), exist_ok=True)
        weights_path= path['WEIGHTS_PATH']
        with open(weights_path,'w') as f:
            json.dump(class_weights,f,indent=2)

        logging.info(f"Class weights for the loss function: {class_weights}")
    

    

        #calculate soft capping weights 
        counts= df['Patient ID'].value_counts()
        df['patient_weight']= df['Patient ID'].map(lambda x: 1.0 / counts[x])

        #create high volume tag for stratfication 
        df['is_high_volume']= (df['Patient ID'].map(counts) > 10).astype(int)

        

        # NEW: Convert View Position to binary (0/1) 
        # Requirement: Prevent the ViT from using camera angle as a proxy for disease.
        # AP (Portable/Emergency) = 1, PA (Standard/Outpatient) = 0
        df['View Position'] = (df['View Position']=='AP').astype(int)
        logging.info("View Position binarized (AP=1, PA=0) to prevent clinical shortcut bias.")

        os.makedirs(output_path,exist_ok=True)
        clinical_data= df.to_csv(os.path.join(output_path,'clinical_data.csv'),index=False)
        logging.info(f"Saved processed clinical CSV at {os.path.join(output_path,'clinical_data.csv')}")

        return df


    except Exception as e:
        raise HEPException(e,sys)