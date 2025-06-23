from huggingface_hub import HfApi, login
import os

# Authentification
login(token=os.environ["HF_TOKEN"])

# Initialiser l'API
api = HfApi()

# Chemin du dossier à uploader
folder_path = "hf_deployment"

# Upload vers Hugging Face
api.upload_folder(
    folder_path=folder_path,
    repo_id="Hiroshi99/Model_SN_GitHub",
    repo_type="model",
    commit_message="Déploiement automatique depuis GitHub Actions"
)