import os
from PIL import Image
from PIL import ImageFile

# Allow Pillow to load truncated images (matches TensorFlow's leniency)
ImageFile.LOAD_TRUNCATED_IMAGES = False

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

clean_dataset("Images/train")
clean_dataset("Images/test")

print("Dataset cleanup complete! You can now run your model.")