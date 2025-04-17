import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

class DataIngestion:

    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_filename"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion started with {self.bucket_name} and {self.file_name}")

    def dowmload_csv_from_gcp(self):
        """
        Downloads a CSV file from Google Cloud Storage and saves it locally.
        """
        try:
            logger.info(f"Credentials path: {CREDENTIALS_PATH}")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV file is sucesfully download to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error("Error while downloading the CSV file.")
            raise CustomException("Failed to download CSV file", e)
    
    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        try:
            logger.info("Starting the splitting process.")
            df = pd.read_csv(RAW_FILE_PATH)
            train_df, test_df = train_test_split(df, test_size=1-self.train_test_ratio, random_state=42)

            train_df.to_csv(TRAIN_FILE_PATH)
            test_df.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error("Error while splitting the data.")
            raise CustomException("Failed to split data into training and test sets ", e)
        
    def run(self):
        """
        Runs the data ingestion process.
        """
        try:
            logger.info("Starting the data ingestion process.")
            self.dowmload_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully.")
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")

        finally:
            logger.info("Data ingestion completed.")

if __name__ == "__main__":
    
    data_ingestion = DataIngestion(read_yaml(COMFIG_PATH))
    data_ingestion.run()

        