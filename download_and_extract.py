import os
import medmnist
from medmnist import INFO
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DATASET_NAME = 'bloodmnist'
SAVE_DIR = "dataset/bloodmnist"

def download_and_extract():
    print(f"Downloading and extracting {DATASET_NAME}...")
    
    info = INFO[DATASET_NAME]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load raw data (will download if not present)
    train_dataset = DataClass(split='train', download=True)
    val_dataset = DataClass(split='val', download=True)
    test_dataset = DataClass(split='test', download=True)
    
    splits = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    label_dict = info['label']
    
    for split_name, dataset in splits.items():
        print(f"Extracting {split_name} split...")
        for i in tqdm(range(len(dataset))):
            img, label_idx = dataset[i]
            # label is a list like [0]
            label = str(label_idx[0])
            class_name = label_dict[label]
            
            # Create directory for this class
            class_dir = os.path.join(SAVE_DIR, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save image
            img_path = os.path.join(class_dir, f"{split_name}_{i}.png")
            img.save(img_path)
            
    print(f"\n✅ Dataset successfully extracted to: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    download_and_extract()
