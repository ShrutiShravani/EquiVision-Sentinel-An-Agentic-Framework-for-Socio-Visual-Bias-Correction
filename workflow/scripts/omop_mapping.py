from logger import logging
from exception import HEPException
import os,sys

def omop_mapping(master_df,output_dir):
    try:
        logging.info("OMOP cmd mapping")
        os.makedirs(output_dir,exist_ok=True)

        gender_mapping= {'M': 8507, 'F': 8532}
        disease_omop_map = {
        'Atelectasis': 4146231,
        'Cardiomegaly': 315286,
        'Effusion': 254061,
        'Infiltration': 4043731,
        'Mass': 432551,
        'Nodule': 4150821,
        'Pneumonia': 255848,
        'Pneumothorax': 255573,
        'Consolidation': 4292193,
        'Edema': 435017,
        'Emphysema': 256441,
        'Fibrosis': 258780,
        'Pleural_Thickening': 4053006,
        'Hernia': 201826
         }
      
        master_df['gender_concept_id']= master_df['Patient Gender'].map(gender_mapping)
        for label,disease_concept_id in disease_omop_map.items():
            master_df[disease_concept_id] = master_df['Finding Labels'].apply(lambda x:1 if label in x else 0)

        #drop old finding
        master_df= master_df.drop(columns=['Finding Labels'])
        logging.info("OMOP mapping successful. Columns are now standardized IDs.")
        return master_df
    except Exception as e:
        raise HEPException(e,sys)