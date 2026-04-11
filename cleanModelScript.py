import os
from PIL import Image
from PIL import ImageFile

# Allow Pillow to load truncated images (matches TensorFlow's leniency)
ImageFile.LOAD_TRUNCATED_IMAGES = False

TRAIN_DIR = 'Images/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/train'
TEST_DIR = 'Images/datasets/tristanzhang32/ai-generated-images-vs-real-images/versions/2/test'

for path in [TRAIN_DIR, TEST_DIR]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    else:
        classes = os.listdir(path)
        print(f"Found: {path} — classes: {classes}")

def clean_dataset(directory):
    bad_files = 0
    print(f"Scanning directory: {directory}")
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    img.load()  # with block ensures file handle is released after
                
            except Exception as e:
                print(f"Corrupted file found and deleted: {file_path} | Reason: {e}")
                try:
                    os.remove(file_path)
                    bad_files += 1
                except PermissionError as pe:
                    print(f"  Could not delete (in use): {file_path} | {pe}")
                
    print(f"Finished scanning {directory}. Deleted {bad_files} bad files.\n")

clean_dataset(TRAIN_DIR)
clean_dataset(TEST_DIR)

print("Dataset cleanup complete! You can now run your model.")