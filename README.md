# Systematic Empirical Analysis of Mixed Precision Training

This repository implements an empirical research pipeline evaluating the impact of Automatic Mixed Precision (AMP) inside data-centric workflows. It is specifically designed to measure how reduced mathematical precision behaves when datasets are messy or mislabeled, particularly within resource-constrained hardware environments (such as consumer laptops lacking dedicated GPUs).

---

## 🧬 1. The Dataset: BloodMNIST
The **MedMNIST (BloodMNIST)** dataset serves as our foundational baseline. Medical imaging inherently requires a data-centric approach, making this an ideal dataset to test our pipeline.

* **Why BloodMNIST?** It contains **17,092 images** of normal peripheral blood cells categorized into 8 distinct classes. 
* **The Visual Challenge:** Blood cells can often look pixelated and blurry at their native 28x28 resolution. To improve the learning capability of our model, we upscale the raw images to `64x64` using soft bicubic interpolation and apply anti-aliasing (Gaussian Blur). This encourages the model to learn actual biological shapes rather than pixel artifacts.

---

## 🏗️ 2. The Architecture: `BloodMNISTResNet`
Traditional architectures like standard `resnet18` or `resnet50` are built for large 224x224 images. Passing a 28x28 blood cell image into a native ResNet results in immediate downsampling that destroys critical features before the network can analyze them.

To solve this, we designed a custom `BloodMNISTResNet` (located in `model.py`):
- **Spatial Preservation:** Instead of a large initial filter, we use a gentle 3x3 filter (`stride=1`). This acts like a magnifying glass, ensuring that the 64x64 blood cell geometry is preserved safely as it enters the model.
- **The Core Engine:** The bulk of the model consists of 16 convolutional layers grouped into 4 blocks. Using skip connections, the network retains the original cellular shape as it analyzes the data.
- **Final Classification:** An Adaptive Pooling layer condenses the features into a single block that outputs the model's confidence across the 8 blood cell types.
- **Resource Constraints:** This custom **18-layer framework** contains **~11.1 Million parameters**. It is balanced to be deep enough for complex features, yet lightweight enough to execute smoothly without running out of standard RAM.

---

## 🧠 3. Training Dynamics & Pipeline
During the 50-epoch experiment, the network executes a refined training loop:

1. **The Forward Pass:** The model evaluates a batch of 128 blood cell images and makes its predictions.
2. **Loss Calculation:** These predictions are compared against the actual medical labels to calculate the error (Loss).
3. **Backpropagation:** The optimizer (`Adam`) adjusts the model weights to improve predictions for the next round.
4. **Best Model Validation (Checkpointing):** To ensure we capture the absolute best version of the model, we evaluate the entire dataset at the end of *every single epoch*. If the network performs better on the validation data than its previous best, it saves a physical checkpoint (`model_amp_best.pth` and `model_fp32_best.pth`). This prevents us from losing peak performance to overtraining.

---

## 📊 4. Empirical Results & Performance

The `experiment.py` script ran a structured evaluation configured for a **128 Batch Size** over **50 Epochs**. We expanded our metrics beyond standard Accuracy to track **Macro F1** and **Precision** scores cleanly using `scikit-learn`:

### Benchmark Table

```text
========================= FINAL REPORT =========================
Dataset: bloodmnist | Batch Size: 128 | Epochs: 50
------------------------------------------------------------------
Scenario        | Mode  | VRAM(MB)  | Time(s)  | Acc(%) | F1     | Prec  
------------------------------------------------------------------
Clean Data      | FP32  | 2559.79   | 2408.10  | 94.86  | 0.9430 | 0.9376
Clean Data      | AMP   | 1492.55   | 1779.66  | 94.86  | 0.9438 | 0.9391
```

### Analysis & Conclusions
The data successfully proves the core benefits of Mixed Precision in our data-centric workflow:
1. **Hardware VRAM Optimization:** Running 16-bit tensors dropped the peak RAM footprint from **~2,560 MB to ~1,492 MB**. This **~41% reduction in memory overhead** is crucial for scaling batch sizes smoothly on constrained hardware.
2. **Computational Acceleration:** Automatic Mixed Precision (AMP) finished the 100-cycle operation in 1779 seconds instead of 2408 seconds, representing a highly efficient **26% speed boost**.
3. **Maintaining High Fidelity:** Moving from FP32 to AMP did not harm the model's intelligence. Both models achieved a solid `94.86%` accuracy, with AMP slightly edging out in `Macro F1` (0.9438 vs 0.9430). Mixed Precision proved it can accelerate training drastically without causing intelligence loss.

---

## 🎯 5. Model Diagnostics & The Granulocyte Challenge
To properly evaluate *why* the model capped at ~95% accuracy, we utilized our auto-generated confusion matrix heatmaps. The diagnostic mapping reveals an interesting biological hurdle rather than a structural failure:

* **Outstanding Success:** The model is almost perfectly mapping standard classes. It correctly identified every single `Lymphocyte` (122/122) and `Platelet` (235/235) with 100% mastery. It was also highly successful on `Eosinophils` (99% accuracy).
* **The Weak Link:** The model struggles specifically with `Immature Granulocytes`. Out of 290 actual immature granulocytes in the test set, the network incorrectly guessed that roughly 40 of them were mature Basophils, Monocytes, or Neutrophils. 

**Conclusion:** Simply training for more epochs will not solve this accuracy cap; doing so would likely just cause simple overfitting. Because the immature granulocyte biologically mimics mature blood cells very closely, the fix needs to be purely data-centric. Future improvements should focus on targeted data augmentation (like Contrast Limited Adaptive Histogram Equalization, or CLAHE, to highlight granules) or penalizing the loss function specifically using "Class Weights" to softly force the network to focus on those immature cells.

---

## 🔬 6. Top-Tier Research Upgrades
Several publication-grade features are natively integrated into `experiment.py`:

1.  **Visual Confusion Matrices:** Medical machine learning demands clear breakdowns of class confusion. Using `seaborn`, the script natively auto-generates beautiful high-resolution heatmap `.png` files at the end of every run.
2.  **Robust Data Augmentation:** To effectively expand the baseline dataset, the `train_loader` incorporates random rotations and dynamic horizontal flipping. This encourages the network to generalize robustly and prevents rote memorization.
3.  **Live Visual Tracking Ecosystem:** Training loop progress is displayed cleanly using `tqdm`, dynamically charting the active epoch, live accuracy, and best checkpoint high-water mark.
4.  **Hardware Dependency Locking:** We've frozen all environmental requirements tightly into a standard `requirements.txt` file, guaranteeing exact academic replication for future setups.

---

## ⚙️ 7. Hardware Orchestration & Scale

When reproducing this analysis locally, be aware of the computational lift required:
*   **Epoch-Level Validation:** The model halts continuous training every epoch to evaluate thousands of images statically. This guarantees we intercept the best model state, but functionally doubles the calculation time per cycle.
*   **The Mathematics of Up-Scaling:** By stretching images computationally from 28x28 to 64x64, the neural network processes roughly **5.2x more computations per image**. 
*   Because of these upgrades, the fully augmented processing pipeline natively takes **30 to 45 minutes** depending on your local hardware acceleration capabilities.

---

## 🚀 8. How to Re-Run the Research

1. **Install Strict Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Trigger the Final Pipeline:**
   ```bash
   python experiment.py
   ```
