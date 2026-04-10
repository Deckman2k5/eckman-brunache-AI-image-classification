import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "G:/ImageClassificationDataset/Images"

path = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")

print(f"Dataset downloaded to: {path}")