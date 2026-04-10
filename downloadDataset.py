import kagglehub
import os

# Change the file location here to wherever you want to download it
os.environ["KAGGLEHUB_CACHE"] = "G:/ImageClassificationDataset/Images"

path = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")

print(f"Dataset downloaded to: {path}")
