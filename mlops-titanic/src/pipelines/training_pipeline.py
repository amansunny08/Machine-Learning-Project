from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    print(train_data)
    print(test_data)