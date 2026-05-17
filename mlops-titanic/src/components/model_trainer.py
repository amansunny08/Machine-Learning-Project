import os
import sys
import pickle

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.utils.logger import logger
from src.utils.mlflow_helper import log_model_mlflow


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test):

        try:
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier()
            }

            report = {}

            for name, model in models.items():

                logger.info(f"Training model: {name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = accuracy_score(y_test, y_pred)

                report[name] = score

                # 🔥 MLflow logging
                log_model_mlflow(
                    model=model,
                    model_name=name,
                    accuracy=score,
                    params=model.get_params()
                )

                logger.info(f"{name} accuracy: {score}")

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):

        try:
            logger.info("Model training started")

            report = self.evaluate_models(
                X_train, y_train, X_test, y_test
            )

            # Select best model
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]

            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Best score: {best_model_score}")

            # Safety check
            if best_model_score < 0.6:
                raise Exception("No good model found (accuracy < 0.6)")

            # Recreate best model
            model_mapping = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier()
            }

            best_model = model_mapping[best_model_name]

            best_model.fit(X_train, y_train)

            # Ensure artifact directory exists
            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path),
                exist_ok=True
            )

            # Save model
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            logger.info("Best model saved successfully")

            return self.config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)