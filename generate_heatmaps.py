import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

from experiment import get_medmnist_dataloaders
from model import BloodMNISTResNet

def generate_heatmap_for_checkpoint(checkpoint_path, save_name, device='cuda'):
    print(f"Generating retroactive heatmap for {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        print(f"[!] Warning: Could not find {checkpoint_path}")
        return

    # Load architecture & weights
    _, val_loader, _, n_classes, class_names = get_medmnist_dataloaders(data_flag="bloodmnist", batch_size=128)
    model = BloodMNISTResNet(num_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
            
    # Draw Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Retroactive Output: {save_name}")
    plt.ylabel('Actual Medical Class')
    plt.xlabel('AI Predicted Class')
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300)
    plt.close()
    print(f"✅ Saved perfectly to {save_name}.png")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Target the physical weights saved exactly from the previous run
    generate_heatmap_for_checkpoint('model_fp32_best.pth', "Previous_Run_FP32_Confusion_Matrix", device)
    generate_heatmap_for_checkpoint('model_amp_best.pth', "Previous_Run_AMP_Confusion_Matrix", device)
