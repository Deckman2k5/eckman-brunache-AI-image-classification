import kagglehub
import os

# Change the file location here to wherever you want to download it
os.environ["KAGGLEHUB_CACHE"] = "C:/Users/ZaneZ/Downloads/py/eckman-brunache-AI-image-classification"

path = kagglehub.dataset_download("tristanzhang32/ai-generated-images-vs-real-images")

print(f"Dataset downloaded to: {path}")
