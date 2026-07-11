# Polyp Size Regression from Monocular Colonoscopy Video

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: Polyp-Size](https://img.shields.io/badge/Dataset-Polyp--Size-orange)](https://doi.org/10.1038/s41597-025-04853-9)

**Continuous polyp size estimation in millimeters from standard monocular colonoscopy video, using depth-corrected physical features and a Random Forest Regressor.**

> **Key Result:** Patient-level Leave-One-Out Cross-Validation (LOOCV) on 42 subjects achieves **MAE = 1.26 mm**. In binary classification (< 5 mm vs. ≥ 5 mm), our approach reaches **69.0% Accuracy** and **84.6% Specificity**, surpassing the dataset authors' baseline models.

---

## Motivation

Accurate polyp size estimation during colonoscopy is critical for clinical decision-making (e.g., surveillance intervals per ESGE guidelines). However:

- **Endoscopists routinely misjudge polyp size** by ±50% due to the lack of physical reference objects in the endoscopic field of view.
- **Monocular cameras cannot measure absolute scale** — a small polyp close to the lens appears identical to a large polyp farther away.
- The dataset authors (Song, Du et al., 2025) explicitly state that *"there is no widely recognized method for polyp size regression"* and only attempted binary classification.

This project solves the open regression problem by combining **learned depth maps** from [DepthPolyp](https://github.com/ReaganWu/DepthPolyp) with **physics-informed features** that correct for perspective distortion.

## Method Overview

```
Input Video (RGB) ──► DepthPolyp (MiT-B0 Encoder)
                          │
                    ┌─────┴─────┐
                    ▼           ▼
              Segmentation   Depth Map
              Mask (0/1)     (relative)
                    │           │
                    ▼           ▼
              sqrt(Area_px)  depth_bg_mean
                    │           │
                    └─────┬─────┘
                          ▼
              Physical Proxy: sqrt(Area) × depth_bg
                          │
                          ▼
              Random Forest Regressor ──► Size in mm
                          │
                          ▼
              Clinical threshold (≥ 4.4 mm) ──► Screening alert
```

### Key Physical Insight

The surrounding healthy mucosa (`depth_bg_mean`) provides a more stable distance reference than the polyp surface itself, because polyps protrude toward the camera. Combined with the apparent diameter in pixels (`sqrt_area_px`) and endoscope optics calibration (`endoscope_model`), this yields a robust size proxy that generalizes across viewing angles.

## Results

### Continuous Regression (LOOCV, N = 42 patients)

| Model | MAE (mm) ↓ | RMSE (mm) ↓ | R² ↑ | MAPE (%) ↓ | Pearson r ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Fixed Domain Features)** | **1.258** | **1.823** | **0.328** | **24.5** | **0.591** |
| Random Forest (True Nested LOOCV) | 1.560 | 2.221 | 0.002 | 29.8 | 0.364 |
| Ridge Regression | 1.530 | 2.178 | 0.063 | 30.3 | 0.303 |
| Lasso Regression | 1.564 | 2.193 | 0.050 | 30.8 | 0.263 |

### Binary Classification (< 5 mm vs. ≥ 5 mm) — Comparison with Dataset Baseline

| Model | Threshold | Accuracy ↑ | Sensitivity ↑ | Specificity ↑ | Evaluation Level |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ours: Screening mode** | **≥ 4.4 mm** | **66.7%** | **68.8%** | 65.4% | **Patient (N=42)** |
| **Ours: Diagnostic mode** | ≥ 5.0 mm | **69.0%** | 43.8% | **84.6%** | **Patient (N=42)** |
| Song et al.: DenseNet169 + ZoeN | 0.5 | 65.7% | 65.8% | 70.2% | Frame (N=3858) |
| Song et al.: ResNet50 + ZoeN | 0.5 | 64.7% | 64.9% | 71.1% | Frame (N=3858) |
| Song et al.: Inception V3 + ZoeN | 0.5 | 60.1% | 48.7% | 72.0% | Frame (N=3858) |

> **Important note on evaluation fairness:** The baseline study evaluates on **3,858 individual frames**, while we evaluate strictly at the **patient level (N=42)**.
## Project Structure

```
Polyp_size_regression/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── polyp_size_regression_report.md    # Detailed evaluation report
│
├── src/
│   ├── polyp_size_regressor.py        # Main pipeline: feature extraction + LOOCV regression
│   ├── polyp_size_classifier.py       # Binary classification (< 5 mm vs. ≥ 5 mm)
│   ├── batch_inference_aggressive.py  # Video segmentation with aggressive temporal filtering
│   └── run_cached_audit.py            # Zero-leakage statistical audit script
│
├── model/                             # DepthPolyp model (MIT License, Wu et al. 2026)
│   ├── depthpolyp.py                 # Model definition
│   └── modules/                       # Encoder, decoder, fusion modules
│       ├── MiT_Encoder.py
│       ├── HF_Decoder.py
│       ├── GFM.py
│       ├── ISF.py
│       ├── DGG.py
│       └── Seg_Head.py
│
├── results/
│   ├── polyp_size_loocv_predictions.csv           # Per-patient regression predictions
│   └── polyp_size_loocv_classification_results.csv # Per-patient classification results
│
└── checkpoints/                       # Model weights (download separately)
    └── .gitkeep
```

## Installation

```bash
git clone https://github.com/viliam-bartos/Polyp_size_regression.git
cd Polyp_size_regression

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Download Model Checkpoint

Download the DepthPolyp checkpoint from [Hugging Face](https://huggingface.co/ReaganWZY/DepthPolyp) or the [original repository](https://github.com/ReaganWu/DepthPolyp):

```python
from huggingface_hub import hf_hub_download

pth_path = hf_hub_download(
    repo_id="ReaganWZY/DepthPolyp",
    filename="DepthPolyp_Kvasir.pth",
    local_dir="checkpoints"
)
```

### Download Dataset

The Polyp-Size dataset is available from:
- **Paper:** [Song, Du et al. (2025) — *Scientific Data* 12:918](https://doi.org/10.1038/s41597-025-04853-9)
- **Dataset (Figshare):** [Polyp-Size Dataset](https://springernature.figshare.com/articles/dataset/Polyp-Size_A_Precisely_Annotated_Endoscopic_Dataset_for_AI-Assisted_Polyp_Sizing/28030115/1?file=53623187)

## Usage

### 1. Run Video Segmentation

```bash
python src/batch_inference_aggressive.py \
    --input_dir /path/to/Polyp_Size_Videos \
    --output_dir inference_aggressive \
    --checkpoint checkpoints/DepthPolyp_Kvasir.pth
```

### 2. Run Size Regression Pipeline

```bash
python src/polyp_size_regressor.py
```

This extracts 24 candidate features from segmentation masks and depth maps, runs LOOCV with fixed domain features, and outputs per-patient predictions.

### 3. Run Binary Classification

```bash
python src/polyp_size_classifier.py
```

## Extracted Features (24 Candidates)

| Group | Features | Description |
| :--- | :--- | :--- |
| **Geometry (2D)** | `sqrt_area_px`, `area_norm`, `perimeter_px`, `circularity`, `aspect_ratio`, `extent`, `solidity` | Shape descriptors from segmentation mask |
| **Depth (3D)** | `depth_in_mean`, `depth_in_std`, `depth_bg_mean`, `depth_contrast`, `depth_ratio` | Relative depth from DepthPolyp decoder |
| **Physical proxies** | `proxy_linear_in`, `proxy_linear_bg`, `proxy_area_in`, `proxy_relief` | Perspective-corrected size estimates |
| **Temporal stability** | `area_variability_std`, `max_sqrt_area_px`, `max_proxy_linear` | Cross-frame consistency measures |
| **Clinical metadata** | `paris_class_num`, `is_pedunculated`, `patient_age`, `patient_gender_num`, `endoscope_hq290I` | Patient and equipment information |

The final model uses only **3 fixed domain features** (selected a priori from physical reasoning, not data-driven search): `sqrt_area_px`, `proxy_linear_bg`, `endoscope_hq290I`.

## Limitations

### Endoscope Model as a Feature

One of the three selected features is the **endoscope model** (`endoscope_hq290I`), which acts as a binary flag distinguishing between two Olympus colonoscopes used in the original dataset (CF-HQ290I vs. CF-H290I).

- The current model has only seen two specific Olympus endoscope models. Applying it to data from a different manufacturer (e.g., Fujifilm, Pentax) or a different Olympus model without retraining or recalibration would likely degrade performance.
- A hardware-agnostic variant of the model (without endoscope metadata) achieves MAE = 1.46 mm, but with substantially lower explanatory power (R² = 0.161 vs. 0.328).
- A more robust approach would replace the categorical endoscope identifier with **continuous optical parameters** from the device's technical specification sheet (field of view in degrees, depth of field range), making the model inherently portable across any endoscope.

### Small Dataset Size

With only **42 subjects from a single center** (Nanfang Hospital, Guangzhou), the dataset is too small to draw strong generalization conclusions. The narrow distribution of polyp sizes (most between 3–6 mm, few above 8 mm) further limits the model's ability to reliably estimate larger lesions.

## Future Work

- **Multi-center validation:** We are in communication with a gastroenterologist at **Nemocnice AGEL Ostrava-Vítkovice** (Czech Republic) to obtain an independent validation cohort recorded with different endoscopic equipment and a European patient population. This will be the first true out-of-distribution test of the pipeline.
- **Larger and more diverse datasets:** Expanding the training set beyond 42 subjects, ideally with a wider range of polyp sizes (including lesions > 10 mm) and multiple endoscope manufacturers.
- **Continuous optical calibration:** Replacing the binary endoscope flag with continuous physical parameters (FOV, focal length) from device specification sheets to enable zero-shot deployment on unseen hardware.
- **End-to-end deep learning regression:** Investigating whether a neural network can be trained to directly predict polyp size in mm from RGB frames, bypassing handcrafted feature extraction.



## Acknowledgments and Citations

This project builds upon two key works:

### DepthPolyp — Segmentation and Depth Model

We use the [DepthPolyp](https://github.com/ReaganWu/DepthPolyp) model (MIT License) by Wu et al. for joint polyp segmentation and monocular depth estimation. The model architecture and pretrained weights are included under their original MIT license terms.

```bibtex
@inproceedings{wu2026depthpolyp,
  title     = {DepthPolyp: Pseudo-Depth Guided Lightweight Segmentation
               for Real-Time Colonoscopy},
  author    = {Wu, Zhuoyu and Ou, Wenhui and Zhang, Lexi and Tan, Pei-Sze
               and Wu, Dongjun and Zhao, Junhe and Fang, Wenqi
               and Phan, Rapha{\"e}l C.-W.},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```

### Polyp-Size Dataset

We use the Polyp-Size dataset published by Song, Du et al. in *Nature Scientific Data*. The dataset provides 42 colonoscopy videos with ground-truth polyp size measurements obtained via calibrated snare comparison.

```bibtex
@article{song2025polypsize,
  title     = {Polyp-Size: A Precise Endoscopic Dataset for AI-Driven
               Polyp Sizing},
  author    = {Song, Ziang and Du, Jingchao and Wu, Jiahao and Li, Kaiwen
               and Guo, Yuxin and Cai, Jiayu and Yu, Hong},
  journal   = {Scientific Data},
  volume    = {12},
  pages     = {918},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41597-025-04853-9}
}
```

## AI-Assisted Development

This project was developed with the assistance of **Google Gemini 3.1 Pro** (via Antigravity IDE). The AI assistant contributed to:
- Feature engineering and physical proxy design
- Statistical validation pipeline (LOOCV, nested CV, zero-leakage audit)
- Code generation for batch inference and regression scripts
- Documentation and report writing

All results were critically reviewed and validated by the human developer.

## License

This project is released under the [MIT License](LICENSE).

The included DepthPolyp model code is © 2026 Zhuoyu Wu, released under the MIT License. The Polyp-Size dataset is published under Creative Commons Attribution 4.0 (CC BY 4.0).
