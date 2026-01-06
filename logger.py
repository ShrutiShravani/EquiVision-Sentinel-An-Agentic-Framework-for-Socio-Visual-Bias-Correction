import logging
import os
from datetime import datetime
import os

LOG_DIR=os.path.join(os.getcwd(),"logs")
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE)


logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)