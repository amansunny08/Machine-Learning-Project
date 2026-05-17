import os
import sys

from dataclasses import dataclass

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.utils.logger import logger


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logger.info("Entered the data ingestion method")

        try:
            # Load inbuilt Titanic dataset
            df = sns.load_dataset("titanic")

            logger.info("Dataset loaded successfully")

            # Create artifacts folder
            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logger.info("Raw dataset saved")

            # Train test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logger.info("Train test split completed")

            # Save train and test data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info("Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)