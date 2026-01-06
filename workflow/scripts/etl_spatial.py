import numpy as np
import pandas as pd
from exception import HEPException
from logger import logging
import os,sys

def run_etl_spatial(input_path,output_path):
    try:
        #handling some non numeric valyues in adi_staternk column
        try:
            logging.info("---Starting spatial etl pipeline---")
            df =pd.read_csv(input_path)
            initial_count = len(df)
            
            df['ADI_STATERNK']=pd.to_numeric(df['ADI_STATERNK'],errors='coerce')

            wi_state_median= df['ADI_STATERNK'].median()

            #fill the non numeri cvalues with edian values
            df['ADI_STATERNK'] = df['ADI_STATERNK'].fillna(wi_state_median)
        
             
            os.makedirs(output_path,exist_ok=True)
            spatial_data= df.to_csv(os.path.join(output_path,'spatial_data.csv'),index=False)
           
            return spatial_data,wi_state_median
        except Exception as e:
            raise HEPException(e,sys) from e

    except Exception as e:
        raise HEPException(e,sys) from e
