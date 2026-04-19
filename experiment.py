import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# MedMNIST library
import medmnist
from medmnist import INFO

# ==========================================
# 1. DATA-CENTRIC DATASET SETUP
# ==========================================
def get_medmnist_dataloaders(data_flag="bloodmnist", batch_size=128, noise_rate=0.0):
    """
    Downloads the dataset, applies data-centric transformations (like label noise),
    and returns DataLoaders.
    """
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Preprocessing: Added explicit anti-aliasing to smooth pixelated visual noise
    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),                 # Interpolates to a higher resolution
        transforms.ToTensor(),                       
        transforms.GaussianBlur(kernel_size=3),      # Mild denoising pass
        transforms.Normalize(mean=[.5] * n_channels, std=[.5] * n_channels)
    ])
    
    # Load raw data
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    
    # ----------------------------------------------------
    # DATA-CENTRIC OPERATION: Simulate noisy annotations
    # ----------------------------------------------------
    if noise_rate > 0.0:
        print(f"[Data-Centric] Simulating noisy labels with {noise_rate*100}% corruption rate...")
        # MedMNIST labels are usually (N, 1). We flatten them.
        train_labels = [label[0] for label in train_dataset.labels]
        num_train = len(train_labels)
        num_noise = int(noise_rate * num_train)
        
        # Select random indices to corrupt
        noise_indices = np.random.choice(num_train, num_noise, replace=False)
        for idx in noise_indices:
            # Assign a random incorrect class
            original = train_labels[idx]
            classes = list(range(n_classes))
            classes.remove(original)
            train_dataset.labels[idx][0] = np.random.choice(classes)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, n_channels, n_classes

from model import BloodMNISTResNet

# ==========================================
# 3. TRAINING REGIME (FP32 vs AMP FP16)
# ==========================================
def train_model(train_loader, val_loader, model, use_amp=False, epochs=5, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Mixed precision scaler (No-op if use_amp=False)
    # Mixed precision scaler (Only for CUDA FP16, not needed for CPU bf16)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == 'cuda' else None
    
    # Memory profiling metrics
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    start_time = time.time()
    
    # --- SIMPLE FEATURE ADDITION: BEST MODEL CHECKPOINTING ---
    best_accuracy = 0.0
    from copy import deepcopy
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # MedMNIST targets shape is [batch, 1], squash to [batch]
            targets = targets.squeeze()
            
            optimizer.zero_grad()
            
            # --- MIXED PRECISION CONTEXT ---
            if use_amp:
                # Use CUDA with float16 or CPU with bfloat16
                amp_device = 'cuda' if device == 'cuda' else 'cpu'
                amp_dtype = torch.float16 if device == 'cuda' else torch.bfloat16
                
                with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                if device == 'cuda' and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                # --- STANDARD FP32 MULTIPLICATION ---
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            
        # --- SIMPLE FEATURE: EVALUATE PER EPOCH AND SAVE BEST ---
        model.eval()
        epoch_correct = 0
        epoch_total = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_targets = val_targets.squeeze()
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs.data, 1)
                epoch_total += val_targets.size(0)
                epoch_correct += (val_predicted == val_targets).sum().item()
        
        epoch_acc = 100 * epoch_correct / epoch_total
        if epoch_acc >= best_accuracy:
            best_accuracy = epoch_acc
            best_model_state = deepcopy(model.state_dict())
            
    end_time = time.time()
    
    # Restore best model before finalizing metrics
    if best_model_state:
        model.load_state_dict(best_model_state)
        # Save it physically to disk
        save_name = "model_amp_best.pth" if use_amp else "model_fp32_best.pth"
        torch.save(best_model_state, save_name)
    
    # Calculate Memory
    peak_vram_mb = 0
    if device == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
    # ----------------------------------------------------
    # METRICS EVALUATION: F1, PRECISION, ACCURACY
    # ----------------------------------------------------
    from sklearn.metrics import accuracy_score, f1_score, precision_score
    
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
    accuracy = accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    
    duration = end_time - start_time
    
    return {
        "duration_sec": duration,
        "peak_vram_mb": peak_vram_mb,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision
    }

# ==========================================
# 4. MAIN EXPERIMENT RUNNER
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing empirical analysis on device: {device}")
    if device == "cpu":
        print("Warning: Mixed Precision (AMP) and VRAM profiling require an NVIDIA GPU to show meanigful differences.")
    
    # Config
    DATASET = "bloodmnist"
    BATCH_SIZE = 128  # Lowered batch size to run lighter on CPU
    EPOCHS = 50       # Boosted to 50 epochs for highest possible accuracy
    
    # As requested by user, we remove the noisy label testing scenario entirely.
    # We will only benchmark exactly one clean state.
    scenarios = [
        {"name": "Clean Data", "noise_rate": 0.0}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n--- SCENARIO: {scenario['name']} ---")
        train_loader, val_loader, n_channels, n_classes = get_medmnist_dataloaders(
            data_flag=DATASET, batch_size=BATCH_SIZE, noise_rate=scenario['noise_rate']
        )
        
        # Test A: FP32
        print(">> Running FP32 (Standard) Training...")
        model_fp32 = BloodMNISTResNet(num_classes=n_classes)
        stats_fp32 = train_model(train_loader, val_loader, model_fp32, use_amp=False, epochs=EPOCHS, device=device)
        print(f"   Accuracy: {stats_fp32['accuracy']:.2f}% | F1: {stats_fp32['f1_score']:.4f} | Prec: {stats_fp32['precision']:.4f} | Time: {stats_fp32['duration_sec']:.2f}s | Peak VRAM: {stats_fp32['peak_vram_mb']:.2f}MB")
        
        # Test B: AMP (FP16/bf16)
        print(">> Running AMP (Mixed Precision) Training...")
        model_amp = BloodMNISTResNet(num_classes=n_classes)
        stats_amp = train_model(train_loader, val_loader, model_amp, use_amp=True, epochs=EPOCHS, device=device)
        print(f"   Accuracy: {stats_amp['accuracy']:.2f}% | F1: {stats_amp['f1_score']:.4f} | Prec: {stats_amp['precision']:.4f} | Time: {stats_amp['duration_sec']:.2f}s | Peak VRAM: {stats_amp['peak_vram_mb']:.2f}MB")
        
        results[scenario['name']] = {
            "FP32": stats_fp32,
            "AMP": stats_amp
        }
    
    # Output final summary table
    print("\n========================= FINAL REPORT =========================")
    print(f"Dataset: {DATASET} | Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}")
    print("------------------------------------------------------------------")
    print(f"{'Scenario':<15} | {'Mode':<5} | {'VRAM(MB)':<9} | {'Time(s)':<8} | {'Acc(%)':<6} | {'F1':<6} | {'Prec':<6}")
    print("-" * 66)
    for scenario_name, data in results.items():
        for mode in ["FP32", "AMP"]:
            s = data[mode]
            print(f"{scenario_name:<15} | {mode:<5} | {s['peak_vram_mb']:<9.2f} | {s['duration_sec']:<8.2f} | {s['accuracy']:<6.2f} | {s['f1_score']:<6.4f} | {s['precision']:<6.4f}")
