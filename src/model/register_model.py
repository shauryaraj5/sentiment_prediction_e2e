# register model - This is not allowed in current version of mlflow in dagshub free tier

import json
import mlflow
from src.logger import logging
import os
import dagshub
import shutil
from datetime import datetime

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "shauryaraj5"
# repo_name = "sentiment_prediction_e2e"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/shauryaraj5/sentiment_prediction_e2e.mlflow')
dagshub.init(repo_owner='shauryaraj5', repo_name='sentiment_prediction_e2e', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

# def register_model(model_name: str, model_info: dict):
#     """Register the model to the MLflow Model Registry."""
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
#         # Register the model
#         model_version = mlflow.register_model(model_uri, model_name)
        
#         # Transition the model to "Staging" stage
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging"
#         )
        
#         logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
#     except Exception as e:
#         logging.error('Error during model registration: %s', e)
#         raise


def pseudo_register_model(model_name: str, model_info: dict, base_dir: str = "models"):
    """
    Simulates a local model registry by versioning model artifacts.
    """
    try:
        run_id = model_info.get("run_id", "N/A")
        model_path = model_info.get("model_path", "unknown")

        trained_model_path = os.path.join(base_dir, f"{model_name}.pkl")
        if not os.path.exists(trained_model_path):
            logging.error(f"Model file not found: {trained_model_path}")
            return

        # Create model registry folder
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Determine next version
        existing_versions = [
            d for d in os.listdir(model_dir)
            if d.startswith("v") and os.path.isdir(os.path.join(model_dir, d))
        ]
        next_version = len(existing_versions) + 1
        version_dir = os.path.join(model_dir, f"v{next_version}")
        os.makedirs(version_dir, exist_ok=True)

        # Copy model file only (not whole folder)
        shutil.copy2(trained_model_path, os.path.join(version_dir, "model.pkl"))

        # Create metadata
        metadata = {
            "model_name": model_name,
            "version": next_version,
            "registered_at": datetime.now().isoformat(),
            "run_id": run_id,
            "source_path": model_path
        }
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Refresh staging directory (copy only contents, not folder itself)
        staging_dir = os.path.join(model_dir, "staging")
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)

        shutil.copy2(os.path.join(version_dir, "model.pkl"), os.path.join(staging_dir, "model.pkl"))
        shutil.copy2(metadata_path, os.path.join(staging_dir, "metadata.json"))

        logging.info(f"✅ Model '{model_name}' registered successfully as version v{next_version}")
        return version_dir

    except Exception as e:
        logging.error(f"❌ Error in pseudo model registration: {e}")
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "model_Log_Reg"
        pseudo_register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

