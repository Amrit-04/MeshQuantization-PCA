# 🧠 3D Mesh Quantization and Invariance Analysis using PCA

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Graphics](https://img.shields.io/badge/Domain-3D%20Geometry%20Processing-blueviolet?style=flat)](https://github.com/Amrit-04/MeshQuantization-PCA)
[![PCA](https://img.shields.io/badge/Method-PCA%20Invariance-success?style=flat)](https://github.com/Amrit-04/MeshQuantization-PCA)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 📘 Project Overview

This project implements an advanced pipeline for **3D mesh normalization, quantization, and reconstruction** with an emphasis on **rotation and translation invariance using PCA (Principal Component Analysis)**.  

The system processes `.obj` meshes, normalizes them using multiple strategies, applies uniform or adaptive quantization, reconstructs the geometry, and evaluates reconstruction accuracy using metrics such as **MSE**, **MAE**, and **PSNR**.  

In addition, a **bonus experiment** was conducted to test invariance under random mesh rotations and translations, confirming that PCA normalization maintains geometric consistency across different orientations.

---

## ⚙️ Key Features

| Feature | Description |
|----------|-------------|
| **Normalization Methods** | Min–Max scaling and Unit Sphere normalization with PCA alignment |
| **Quantization Types** | Uniform (256–2048 bins) and Curvature-based Adaptive Quantization |
| **Error Metrics** | MSE, MAE, PSNR |
| **Invariance Testing** | Random rotations/translations validated PCA normalization consistency |
| **Visualization Tools** | Open3D-based viewers for side-by-side, heatmap, and PCA comparisons |

---
## 🧮 Running the Project

### 1️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib trimesh open3d scikit-learn tqdm scipy

2️⃣Run Full Quantization Pipeline
    python src/mesh_quant_full.py

3️⃣ Evaluate Best Combination
    python src/find_best_method.py

4️⃣ Bonus: Rotation and Translation Invariance
    python src/bonus_invariance_adaptive_compare.py


5️⃣ Visualization Options
    Side-by-Side (Original vs Reconstruction)
        python src/visualize_side_by_side.py --mesh girl --pipeline full --method minmax --pca pca --bins 1024

    Error Heatmap
        python src/visualize_with_heatmap.py --mesh talwar --pipeline full --method unit_sphere --pca pca --bins 2048

    PCA vs Unit Sphere Comparison
        python src/visualize_norm_compare.py --mesh girl --rotation 0

📊 Results Summary
| Normalization | PCA | Quantization | Avg MSE | PSNR (dB) | Observation                 |
| ------------- | --- | ------------ | ------- | --------- | --------------------------- |
| Min–Max       | No  | 2048 bins    | ~0.0    | 82.0      | Best numeric accuracy       |
| Unit Sphere   | Yes | 1024 bins    | 1.3e−07 | 79.4      | Stable, rotation-tolerant   |
| PCA           | Yes | Adaptive     | 2.1e−07 | 80.1      | Best under random rotations |

🧠 Bonus Experiment Findings
| Mesh   | Adaptive MSE | Uniform MSE | % Improvement | Observation                              |
| ------ | ------------ | ----------- | ------------- | ---------------------------------------- |
| Girl   | 2.17e−07     | 6.68e−08    | 69% ↓         | PCA maintained shape across orientations |
| Talwar | 2.14e−07     | 6.66e−08    | 69% ↓         | Adaptive preserved fine blade edges      |
| Branch | 1.11e−06     | 3.46e−07    | 69% ↓         | Elongated model benefited most           |
| Table  | 3.32e−07     | 1.04e−07    | 69% ↓         | Flat surfaces remained stable            |


This project demonstrates a full 3D mesh quantization workflow capable of high-fidelity reconstruction and robust performance under spatial transformations.
By leveraging PCA-based normalization, the system achieves rotation and translation invariance, while adaptive quantization enhances accuracy in geometrically detailed areas.
The final results confirm that high-resolution (1024/2048-bin) quantization with PCA alignment ensures minimal information loss and consistent geometry preservation across orientations.
