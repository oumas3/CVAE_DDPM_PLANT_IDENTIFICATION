# Hybrid CVAE–DDPM for Conditional Data Augmentation in Plant Disease Classification

## Overview

This project proposes a hybrid generative data augmentation framework combining a Conditional Variational Autoencoder (CVAE) and a Denoising Diffusion Probabilistic Model (DDPM) to improve plant disease classification performance under limited and imbalanced data conditions.

The system integrates class-conditional synthetic image generation into a supervised classification pipeline using ResNet18. The objective is to evaluate whether combining CVAE and DDPM improves generalization compared to training on real data only.

This work was conducted as part of a Master's research project in Data Science and Artificial Intelligence.

---

## Key Contributions

- Implementation of a class-conditional CVAE for structured latent-space generation.
- Implementation and fine-tuning of a DDPM for higher-fidelity synthetic image generation.
- Hybrid augmentation strategy combining real, CVAE-generated, and DDPM-generated data.
- Controlled evaluation using both discriminative and generative quality metrics.
- Ablation study to measure the isolated and combined impact of generative models.

---

## Dataset

Dataset: PlantVillage  
Total images: ~54,000 RGB images  
Number of classes used: 15  
Image size: 64×64  

Data split:
- 70% Training
- 15% Validation
- 15% Test

Important:
Synthetic images are used only during training.
All performance metrics are computed on real unseen test data.

---

## Methodology

### 1. Conditional Variational Autoencoder (CVAE)

- Latent dimension: 128
- Loss function: Reconstruction Loss + KL Divergence
- Class-conditioned generation via label embedding

The CVAE improves structural diversity within each disease class.

---

### 2. Denoising Diffusion Probabilistic Model (DDPM)

- 50 diffusion steps
- Noise prediction training objective
- Fine-tuned on the PlantVillage dataset

The DDPM produces higher-fidelity images with improved intra-class realism.

---

### 3. Hybrid Augmentation Strategy

Final training set:

X_hybrid = X_real ∪ X_CVAE ∪ X_DDPM

## Classifier

- **Architecture:** ResNet18  
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  

### Hypothesis

- CVAE improves semantic diversity.
- DDPM improves visual realism.
- The combined approach improves classifier robustness and generalization.

---

## Training Configuration

| Model | Batch Size | Epochs | Learning Rate | Optimizer |
|--------|------------|--------|---------------|------------|
| CVAE   | 64         | 100    | 1e-3          | Adam       |
| DDPM   | 16         | 50     | 2e-4          | Adam       |

### Infrastructure

- NVIDIA Tesla T4 (Kaggle environment)
- PyTorch 2.1
- CUDA 12.1

---

## Evaluation Metrics

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Generative Quality Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- FID (Fréchet Inception Distance)

---

## Results

### Baseline (Real Data Only)

- **Accuracy:** 0.9561  
- **F1-score:** 0.955  

### Hybrid (CVAE + DDPM)

- **Accuracy:** 0.968  
- **F1-score:** 0.961  

**Improvement:**  
+1.2% absolute accuracy gain over baseline.

---

## FID Scores

| Model | FID |
|--------|------|
| CVAE  | 203.25 |
| DDPM  | 195.97 |

Although FID values remain relatively high, classification performance improves.  
This suggests that synthetic samples act as a regularization mechanism rather than perfectly replicating the real data distribution.

---

## Ablation Study

| Configuration | Accuracy |
|----------------|-----------|
| Real only | 0.9561 |
| Real + CVAE | 0.961 |
| DDPM only | 0.631 |
| CVAE + DDPM | 0.968 |

The results indicate complementary effects when combining CVAE and DDPM.

---

## Limitations

- FID scores remain high due to domain mismatch.
- Performance improvement is moderate (+1.2%).
- Risk of synthetic bias.
- Dataset contains controlled backgrounds (limited real-world variability).

---

## Future Work

- Evaluation on real-world field images.
- Integration of latent diffusion models.
- Advanced class-balancing strategies.
- Lightweight deployment-ready inference model.
