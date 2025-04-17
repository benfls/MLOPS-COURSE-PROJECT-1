import os 
import pandas 
from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
import yaml

logger = get_logger(__name__)

def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} is not in the given path.")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info("Successfully read the YAML file.")
            return config
        
    except Exception as e:
        logger.error(f"Error reading YAML file.")
        raise CustomException("Failed to read YAML file", e)
    
def load_data(file_path: str) -> pandas.DataFrame:
    """
    Loads data from a CSV file and returns it as a pandas DataFrame.
    """
    try:
        logger.info("Loading data.")
        return pd.read_csv(file_path)    
    except Exception as e:
        logger.error(f"Error loading the data {e}.")
        raise CustomException("Failed to load data", e)