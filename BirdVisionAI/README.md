# BirdVision AI v2

> **Hybrid Bird Species Recognition System** — ResNet-18 · OOD Detection · CLIP Fallback · Grad-CAM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-ff4b4b.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

BirdVision AI is a production-quality, portfolio-grade bird species recognition system built on the CUB-200-2011 dataset. Unlike a plain classifier, it incorporates a **hybrid recognition pipeline** that gracefully handles both known and unknown bird species.

**The core problem it solves:** A vanilla ResNet-18 trained on 200 species always outputs one of those 200 classes — even when the uploaded image shows a crow, pigeon, or peacock that was never in the training set. BirdVision AI detects this and intelligently falls back.

---

## System Architecture

```
User Uploads Image
        │
        ▼
 BirdVisionAI Classifier
  (ResNet-18 · 200 Species · CUB-200-2011)
        │
        ▼
  Confidence Analysis
  (Maximum Softmax Probability)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 Known    Unknown
Species   Species
   │         │
   ▼         ▼
Return    CLIP / OpenCLIP
Result    Fallback Model
   │         │
   ▼         ▼
Grad-CAM  Species
Heatmap   Suggestions
   │
   ▼
Species
Info Panel
```

---

## Features

| Feature | Description |
|---------|-------------|
| **ResNet-18 Classifier** | Fine-tuned on CUB-200-2011, top-5 predictions |
| **OOD Detection** | Maximum Softmax Probability thresholding |
| **CLIP Fallback** | OpenCLIP ViT-B-32 zero-shot species suggestion on unknown images |
| **Grad-CAM** | Visualise which image regions drive the model's prediction |
| **Confidence Chart** | Interactive Plotly bar chart of top-5 scores |
| **Species Info Panel** | Family, habitat, diet, region, conservation status |
| **Multi-page UI** | Classify · Analytics · History |
| **Session Analytics** | Total scans, OOD rate, average confidence |
| **Configurable Thresholds** | Sidebar sliders for real-time tuning |

---

## Dataset

**CUB-200-2011 (Caltech-UCSD Birds-200-2011)**

| Property | Value |
|----------|-------|
| Species | 200 North American bird species |
| Images | 11,788 total |
| Train split | ~5,994 images |
| Test split | ~5,794 images |
| Annotations | Bounding boxes, part locations, attributes |

[Dataset homepage](http://www.vision.caltech.edu/datasets/cub_200_2011/)

---

## Model

| Property | Value |
|----------|-------|
| Architecture | ResNet-18 |
| Pretrained | ImageNet |
| Fine-tuned | CUB-200-2011 (200 classes) |
| Checkpoint | `checkpoints/cub_resnet18.pth` (~44 MB) |
| Input size | 224 × 224 RGB |
| OOD method | Maximum Softmax Probability (Hendrycks & Gimpel, 2017) |

---

## Project Structure

```
BirdVisionAI/
│
├── app/
│   ├── main.py                  # Streamlit entry point & page router
│   │
│   ├── config/
│   │   └── settings.py          # All configurable constants
│   │
│   ├── models/
│   │   ├── model_loader.py      # ResNet-18 checkpoint loader
│   │   ├── predictor.py         # Softmax inference pipeline
│   │   ├── ood_detector.py      # Maximum Softmax Probability OOD
│   │   ├── gradcam.py           # Hook-based Grad-CAM
│   │   └── clip_fallback.py     # OpenCLIP zero-shot fallback
│   │
│   ├── preprocessing/
│   │   └── transforms.py        # CUB-standard image transforms
│   │
│   ├── utils/
│   │   ├── class_loader.py      # classes.txt parser
│   │   ├── logger.py            # Structured logging
│   │   └── helpers.py           # Shared utilities
│   │
│   ├── ui/
│   │   ├── sidebar.py           # Navigation & settings sidebar
│   │   ├── upload.py            # Image upload widget
│   │   ├── results.py           # Top-5 prediction cards
│   │   ├── confidence_chart.py  # Plotly confidence bar chart
│   │   ├── species_info.py      # Species information panel
│   │   ├── templates.py         # Pure HTML template functions
│   │   └── css_loader.py        # CSS injection helper
│   │
│   └── assets/
│       ├── styles.css           # Dark-theme global stylesheet
│       └── bird_icon.svg        # App icon
│
├── checkpoints/
│   └── cub_resnet18.pth         # Trained model weights
│
├── data/
│   ├── classes.txt              # 200 CUB species names
│   └── species_db.json          # Species info database
│
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Installation

### Local (Python venv)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/BirdVisionAI.git
cd BirdVisionAI

# 2. Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install OpenCLIP for the CLIP fallback feature
pip install open_clip_torch

# 5. Run the app
streamlit run app/main.py
```

### Docker

```bash
docker build -t birdvision-ai .
docker run -p 8501:8501 birdvision-ai
# Open http://localhost:8501
```

---

## Configuration

All configurable values live in `app/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_PATH` | `checkpoints/cub_resnet18.pth` | Path to model checkpoint |
| `CLASS_NAMES_PATH` | `data/classes.txt` | Path to species list |
| `OOD_THRESHOLD` | `0.70` | MSP threshold below which bird is "unknown" |
| `IMAGE_SIZE` | `224` | Input image resize dimension |
| `TOP_K` | `5` | Number of top predictions |
| `GRADCAM_ENABLED` | `True` | Show Grad-CAM heatmap by default |
| `GRADCAM_ALPHA` | `0.45` | Heatmap overlay opacity |
| `CLIP_ENABLED` | `True` | Enable CLIP fallback on OOD |
| `CLIP_MODEL` | `ViT-B-32` | OpenCLIP model variant |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | Streamlit |
| Deep learning | PyTorch 2.x + TorchVision |
| Backbone | ResNet-18 |
| OOD detection | Maximum Softmax Probability |
| Explainability | Grad-CAM (custom implementation) |
| CLIP fallback | OpenCLIP (ViT-B-32, OpenAI weights) |
| Visualisation | Plotly |
| Image processing | Pillow, OpenCV (optional) |
| Styling | Custom CSS (dark theme) |

---

## OOD Detection Explained

The **Maximum Softmax Probability (MSP)** method works as follows:

1. Run the image through ResNet-18 → get 200 softmax probabilities
2. Find `max(P)` — the highest confidence score
3. Compare against `OOD_THRESHOLD` (default 70%):
   - `max(P) ≥ 0.70` → **Known species** → return top-5 predictions
   - `max(P) < 0.70` → **Unknown species** → show OOD alert + trigger CLIP

A well-trained classifier concentrates probability mass on the correct class for known species (high MSP). For unseen species (crow, pigeon, peacock), probability mass spreads thinly across all 200 classes, resulting in low MSP.

---

## Grad-CAM Explained

Grad-CAM computes gradient of the predicted class score with respect to the final convolutional feature maps. The magnitude of these gradients indicates how important each spatial region was for the prediction.

```
Target class score → backprop → gradients at layer4[-1].conv2
                                        ↓
                              Global Average Pool
                                        ↓
                              Weighted activation maps
                                        ↓
                              ReLU + Normalize → heatmap
                                        ↓
                              Overlay on original image
```

---

## Future Improvements

- [ ] Temperature scaling for better-calibrated probabilities
- [ ] Energy-based OOD scoring (Liu et al., 2020)
- [ ] BioCLIP integration (biology-specific CLIP model)
- [ ] Fine-grained species comparison view
- [ ] Bird call audio classification
- [ ] Geo-based species probability prior
- [ ] Batch image processing mode
- [ ] REST API with FastAPI backend

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- CUB-200-2011 dataset by Wah et al., Caltech Vision Lab
- Grad-CAM: Selvaraju et al., ICCV 2017
- MSP OOD: Hendrycks & Gimpel, ICLR 2017
- OpenCLIP: LAION AI
