import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging
import wandb


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("SENTIMENT_PREDICTION")
if not dagshub_token:
    raise EnvironmentError("SENTIMENT_PREDICTION environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "shauryaraj5"
repo_name = "sentiment_prediction_e2e"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/shauryaraj5/sentiment_prediction_e2e.mlflow')
# dagshub.init(repo_owner='shauryaraj5', repo_name='sentiment_prediction_e2e', mlflow=True)
# -------------------------------------------------------------------------------------

# Load model from file
def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('✅ Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('❌ File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('❌ Unexpected error occurred while loading the model: %s', e)
        raise

# Load data from CSV
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('✅ Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('❌ Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('❌ Unexpected error occurred while loading the data: %s', e)
        raise

# Evaluate the model
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('✅ Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('❌ Error during model evaluation: %s', e)
        raise

# Save metrics to a JSON file
def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('✅ Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('❌ Error occurred while saving the metrics: %s', e)
        raise

# Save model run ID and path
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info('✅ Model info saved to %s', file_path)
    except Exception as e:
        logging.error('❌ Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("dvc-pipeline-1")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model('./models/model_Log_Reg.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')

            # # Initialize Weights & Biases logging
            # wandb_run = wandb.init(
            #     # Set the wandb entity where your project will be logged (generally your team name).
            #     entity="Shaurya_Projects",
            #     # Set the wandb project where this run will be logged.
            #     project="sentiment_prediction_e2e",
            #     # Track hyperparameters and run metadata.
            #     # config={
            #     #     "model_name": "LogisticRegression",
            #     #     "method": "Bag of Words",
            #     #     "C": 1,
            #     #     'solver': 'liblinear',
            #     #     'penalty': 'l1',
            #     # },
            # )
            # print("✅ Weights & Biases Initialized.")

            # Log metrics to MLflow and wandb
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)    # mlflow log
                #wandb.log({metric_name: metric_value})          # wandb log
            #logging.info("✅ Metrics logged to both MLflow and Weights & Biases.")

            # Log model parameters to MLflow and wandb
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                    #wandb_run.config.update(params)
            #logging.info("✅ Model Parameters logged to both MLflow and Weights & Biases.")

            # Log model to MLflow- Not aplicable anymore
            #mlflow.sklearn.log_model(clf, "model")

            # Log the model as an artifact to wandb
            # model_artifact = wandb_run.Artifact("log_reg_model", type="model")
            # model_artifact.add_file("./models/model_Log_Reg.pkl")
            # wandb.log_artifact(model_artifact)
            
            # Save model info
            save_model_info(run.id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow and wandb
            mlflow.log_artifact('reports/metrics.json')
            #wandb_run.save('reports/metrics.json')
            #logging.info("✅ Evaluation completed and logged to both MLflow and Weights & Biases.")

            #wandb_run.finish()
            logging.info("✅ Evaluation completed and logged to both MLflow and Weights & Biases.")

        except Exception as e:
            logging.error('❌ Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
