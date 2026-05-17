import mlflow
import mlflow.sklearn


def log_model_mlflow(model, model_name, accuracy, params=None):

    mlflow.set_experiment("titanic_mlops")

    with mlflow.start_run():

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)

        if params:
            for key, value in params.items():
                try:
                    mlflow.log_param(key, value)
                except:
                    pass  # ignore non-serializable params

        mlflow.sklearn.log_model(model, "model")

        print(f"Logged {model_name} to MLflow | accuracy: {accuracy}")git s