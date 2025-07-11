import os
from PIL import Image
from io import BytesIO

def is_corrupted(image_path):
    """Check if an image file is corrupted."""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
            # First verify the image can be opened
            img = Image.open(BytesIO(img_data))
            img.verify()  # Verify integrity without loading full image
            
            # Additional check: try to load the image
            img = Image.open(BytesIO(img_data))
            img.load()  # Attempt to load the image data
            return False
    except Exception as e:
        print(f"Corruption detected in {os.path.basename(image_path)}: {str(e)}")
        return True

def clean_dataset(folder_path):
    """Remove corrupted images from a folder."""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    corrupted_count = 0
    valid_count = 0
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip non-image files
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
            
        if is_corrupted(file_path):
            try:
                os.remove(file_path)
                corrupted_count += 1
                print(f"❌ Deleted corrupted: {filename}")
            except Exception as e:
                print(f"Failed to delete {filename}: {str(e)}")
        else:
            valid_count += 1
    
    print(f"\nFolder: {folder_path}")
    print(f"Total corrupted files removed: {corrupted_count}")
    print(f"Valid images remaining: {valid_count}")

# Run on both folders
print("Starting image cleanup...")
#clean_dataset(r"C:\Users\inno\Documents\validation_dataset\maize_dataset")
#clean_dataset(r"C:\Users\inno\Documents\validation_dataset\non_maize_dataset")
clean_dataset(r"C:\Users\inno\Documents\data\Non_Maize")
print("Cleanup complete!")