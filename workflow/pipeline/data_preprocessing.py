from logger import logging
from exception import HEPException
import os,sys
from logger import logging
from exception import HEPException
from workflow.scripts import omop_mapping
from workflow.scripts.etl_clinical import run_clinical_etl
from workflow.scripts.etl_spatial import run_etl_spatial
from workflow.scripts.etl_visual import run_visual_etl_process
from workflow.scripts.bridge_linker import bridge_linkage_engine
from utils.load_config import load_config
import json

path= load_config(config_path="config\path.yaml")

class data_preprocessing():
    def __init__(self,output_dir):
        self.output_dir=path['PROCESSED_PATH']

    def main_preprocessing_pipeline(self):
        try:
            logging.info("Starting master preprocessing pipeline")
            os.makedirs(self.output_dir,exist_ok=True)

            #step 1 clinical etl
            logging.info("Running clinical etl pipeline..")
            processed_clinical_df =run_clinical_etl(input_path=path['CLINICAL_PATH'])
            
            logging.info("Running spatial etl pipeline")
            
            processed_spatial_df= run_etl_spatial(input_path=[path['ADI_PATH']])
            
            #visual etl visual
            png_found = False
            image_root = os.path.join(path['IMAGES_PATH'], "images") # Targeting data/images/images
            png_found=False
            for root,dir,files in os.walk(path['IMAGES_PATH']):
                for file in files:
                    if file.endswith(".png"):
                        png_found=True
                        break
            
            if png_found:
                logging.info("Running visual etl")
                global_mean,global_std= run_visual_etl_process(df=processed_clinical_df,target_size=(224,224))

            stats={
                "global_mean":global_mean,
                "global_std":global_std,
                "target_size": [224, 224],
                "normalization": "div_255"
            }
            stats_path=path['STATS_PATH']
            with open(stats_path,"w") as f:
                json.dump(stats)
            logging.info(f"global normalization saved to {stats_path}")

            logging.info("Running linkage between clinical and spatial data")
            pre_master_df= bridge_linkage_engine(processed_clinical_df,processed_spatial_df)
        
            logging.info("OMOP MAPPING OF GENDER AND DISEASE")
            master_df= omop_mapping(master_df=pre_master_df,output_dir=self.output_dir)

            master_df.to_csv(os.path.join(path['PROCESSED_PATH'],"processed_data.csv"),index=False)

            logging.info("Success: Master Dataset with OMOP Mapping and ADI Context is ready!")

        except Exception as e:
            raise HEPException(e,sys)
