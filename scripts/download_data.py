"""Download training data once. Accept the Kaggle competition rules first."""
from pathlib import Path
from dotenv import load_dotenv
import kagglehub

from recommender_santander.utils import get_project_root

root = get_project_root()
load_dotenv(root / ".env")
folder = root / "data/raw/train"
filename = "train_ver2.csv.zip"
if (folder / filename).exists() or (folder / "train_ver2.csv").exists():
    print("Training data already exists:", folder)
else:
    folder.mkdir(parents=True, exist_ok=True)
    print(kagglehub.competition_download(
        "santander-product-recommendation", path=filename, output_dir=str(folder),
    ))
