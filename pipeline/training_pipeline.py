from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.path_config import *


if __name__ == "__main__":
    ### 1. Data Ingestion
    data_ingestion = DataIngestion(read_yaml(COMFIG_PATH))
    data_ingestion.run()

    ### 2. Data Processing
    procesor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, COMFIG_PATH)
    procesor.process_data()

    ### 3. Model Training
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    trainer.run()