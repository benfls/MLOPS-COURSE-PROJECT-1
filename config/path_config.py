import os

########################################## DATA INGESTION ##########################################

RAW_DIR = "artifacts/raw"
CREDENTIALS_PATH = r"C:\Users\T0317216\Documents\training\Beginner_Advanced_MLOps\credentials.json"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

COMFIG_PATH = "config/config.yaml"


##################################### DATA PROCESSING #####################################
PROCESSED_DIR = "artifacts/processed"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")


########################################### MODEL TRAINING ##########################################
MODEL_OUTPUT_PATH = "artifacts/model/lgbm_model.pkl"
