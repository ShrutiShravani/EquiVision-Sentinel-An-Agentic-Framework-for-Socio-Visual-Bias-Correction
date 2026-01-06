from winsound import SND_PURGE
import pandas as pd
from exception import HEPException
from logger import logging
import os,sys
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

def run_visual_etl_process(df,target_size=(224,224)):
    """
    1. Multi-threaded Path Mapping
    2. Resizing to 224x224
    3. Calculating Global Normalization Constants
    """
    try:

        def process_image(full_path):
            if os.path.exists(full_path):
                #load and resize

                img= cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
                
                #aspect ratio audit (padding instead of streteching)
                h,w= img.shape[:2]
                desired_size= target_size[0]

                #calculate ratio and new dimensions
                ratio= float(desired_size)/max(h,w)
                new_size= tuple([int(x*ratio) for x in (w,h)])  

                img= cv2.resize(img,(new_size[0],new_size[1]))

                #add black padding to make it exactly 224*224
                
                delta_w= desired_size - new_size[0]  
                delta_h= desired_size- new_size[1]
                top,bottom=delta_h//2,delta_h-(delta_h//2)
                left,right= delta_w//2,delta_w-(delta_w//2)

                img= cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
                
                #conevrtign balck n white image to rbg
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return img_rgb/255.0    
            return None
        
        #we sample 2000 images to get global constants
        sample_images= df['abs_image_path'].sample(2000).tolist()

        logging.info("Resizing and calculating Global Normalization constants...")
        
        with ThreadPoolExecutor() as executor:
            results= [res for res in executor.map(process_image,sample_images) if res is not None]

        global_mean= np.mean(results)
        global_std= np.std(results)

        logging.info(f"Global mean :{global_mean:.4f}")
        logging.info(f"Global std :{global_std:.4f}")

        return global_mean, global_std
    
    except Exception as e:
        raise HEPException(e,sys)