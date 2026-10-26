import os
from huggingface_hub import HfApi

def deploy():
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_SPACE_ID", "your-username/nfl-prospect-evaluator")
    
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set.")
        
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["*.csv", ".git*", "__pycache__/*", "data/*"]
    )
    print(f"🚀 Successfully deployed to Hugging Face Space: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    deploy()