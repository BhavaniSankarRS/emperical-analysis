import 
import logging
from PIL import Image, ImageFilter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOURCE_DIR = "dataset/bloodmnist"
TARGET_DIR = "dataset_clean/bloodmnist"
NUM_WORKERS = os.cpu_count() or 4

def process_single_image(args: tuple) -> None:
    """
    Upscales and denoises a single image.
    args: (src_img_path, tgt_img_path)
    """
    src_img_path, tgt_img_path = args
    # 1. Open the raw 28x28 pixelated blob
    img = Image.open(src_img_path).convert('RGB')
    
    # 2. Upscale aggressively to 128x128 using high-quality BICUBIC interpolation
    img_large = img.resize((128, 128), resample=Image.Resampling.BICUBIC)
    
    # 3. Apply a slight smoothing (anti-aliasing/blur) to blend the interpolation
    img_smooth = img_large.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # 4. Save the new clean image
    img_smooth.save(tgt_img_path)

def upscale_and_denoise() -> None:
    """
    Iterates through the dataset and applies high-quality bicubic upscaling and 
    defocus filtering to counter raw low-res pixelation, parallelized across CPU cores.
    """
    logger.info("Initializing physical dataset cleaning process via Multi-Threading...")
    
    if not os.path.exists(SOURCE_DIR):
        logger.error(f"Could not find {SOURCE_DIR}. Run download_and_extract.py first.")
        return

    # Gather tasks
    tasks = []
    for split in os.listdir(SOURCE_DIR):
        split_path = os.path.join(SOURCE_DIR, split)
        if not os.path.isdir(split_path): continue
        
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path): continue
            
            target_class_path = os.path.join(TARGET_DIR, split, class_name)
            os.makedirs(target_class_path, exist_ok=True)
            
            for img_name in os.listdir(class_path):
                if not img_name.endswith('.png'): continue
                src_img_path = os.path.join(class_path, img_name)
                tgt_img_path = os.path.join(target_class_path, img_name)
                tasks.append((src_img_path, tgt_img_path))
                
    # Execute in parallel
    logger.info(f"Registered {len(tasks)} target images. Executing on {NUM_WORKERS} workers.")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Upscaling & Denoising"))
                
    logger.info(f"All images successfully upscaled and cleaned into '{TARGET_DIR}'.")

if __name__ == "__main__":
    upscale_and_denoise()
