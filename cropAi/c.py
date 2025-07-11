import os
from PIL import Image

def check_dataset(directory):
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png')):
            try:
                img = Image.open(os.path.join(directory, filename))
                img.verify()  # Check for corruption
                if img.size != (224, 224):  # Resize if needed
                    print(f"Resizing {filename}...")
                    img = img.resize((224, 224))
                    img.save(os.path.join(directory, filename))
            except Exception as e:
                print(f"❌ Corrupted: {filename} - {e}")
                os.remove(os.path.join(directory, filename))

check_dataset( r"C:\Users\inno\Documents\validation_dataset\non_maize_dataset")
check_dataset( r"C:\Users\inno\Documents\validation_dataset\non_maize_dataset")