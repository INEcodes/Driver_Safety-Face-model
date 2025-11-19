#  Multi-Modal Driver Inattention Detection System

<div align="center">

**Advanced Driver Safety System using Deep Learning & Sensor Fusion**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A research project from VIT Bhopal University's School of Computing Science and Engineering*

</div>

---

## 👥 Research Team

### Contributors
- **Amogh Biradar** - amoghbiradar2504@gmail.com
- **Maan Baria** - maanbaria@outlook.com
- **Abhinav Tiwari** - abhinavtiwaridev@gmail.com
- **Tanmay Verma** - tanmayvr10@gmail.com
- **Abhay Singh Chauhan** - abhayaps05@gmail.com

### Supervisor
- **Dr. Vipin Jain** - er.vipinjain@gmail.com | [ORCID: 0000-0002-0099-3933](https://orcid.org/0000-0002-0099-3933)

---

##  Overview

This repository implements an **Integrated Driver Safety System (IDSS)** that combines multiple sensing modalities and advanced deep learning to detect driver inattention, drowsiness, and distraction. The system is designed to meet SAE Level 3 autonomous driving requirements and provides critical safety monitoring for modern vehicles.

###  Key Features

- ** Classical ML Models**: SVM, KNN, Random Forest, Naive Bayes, Decision Tree
- ** Deep Learning Architectures**: CNN, LSTM, CNN-LSTM hybrids
- ** ASTN Model**: Proposed Attention-Based Spatio-Temporal Network achieving **96.54% accuracy**
- ** Multi-Modal Detection**: Vision-based drowsiness and distraction classification
- ** Real-Time Performance**: YOLOv8n integration for edge deployment
- ** Comprehensive Analysis**: Benchmarked against state-of-the-art commercial systems

---

## Performance Highlights

Our **Proposed ASTN Model** significantly outperforms baseline approaches:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Proposed ASTN** | **96.54%** | **94.44%** | **96.63%** | **95.87%** |
| CNN-LSTM | 89.34% | 87.23% | 87.32% | 86.14% |
| Random Forest | 84.75% | 87.12% | 88.47% | 86.07% |
| CNN | 88.46% | 84.45% | 88.63% | 87.25% |

---

## 📂 Project Structure

```
driver-inattention-detection/
├── 📁 .vscode/              # Development configuration
├── 📁 Codes/                # Implementation notebooks
├── 📁 src/                  # Core source code
│   ├── __init__.py          
│   ├── data_loader.py       # Dataset utilities
│   ├── feature_extractor.py # Feature engineering
│   ├── ml_traditional.py    # Classical ML models
│   ├── models.py            # Deep learning architectures
│   └── train.py             # Training pipelines
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quick Start

### Google Colab Setup

```python
# Install dependencies
!pip install opendatasets tensorflow scikit-learn

# Download dataset
import opendatasets as od
od.download("https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset")

# Clone repository
!git clone https://github.com/yourusername/driver-inattention-detection.git
%cd driver-inattention-detection
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/driver-inattention-detection.git
cd driver-inattention-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage Examples

### Train Classical ML Models

```python
from src.ml_traditional import run_traditional_ml_models

# Train and evaluate SVM, KNN, Random Forest, etc.
results = run_traditional_ml_models()
```

### Train ASTN Model

```python
from src.models import create_proposed_astn_model
from src.train import train_model

# Create Attention-Based Spatio-Temporal Network
model = create_proposed_astn_model(
    sequence_length=15,
    filters=[32, 64, 128, 256],
    lstm_units=[128, 64],
    dropout_rate=0.4
)

# Train the model
history = train_model(model, epochs=175, batch_size=32)
```

### YOLOv8n Real-Time Detection

```python
from src.models import load_yolov8n_model

# Load pre-trained YOLOv8n for distraction detection
model = load_yolov8n_model()
# Classes: safe_driving, talking_phone, texting_phone, turning
```

---

## 🔬 Research Contributions

This project implements the **Integrated Driver Safety System (IDSS)** framework as described in our research paper:

1. **Multi-Modal Fusion Architecture** combining:
   - Vision-based monitoring (NIR/RGB-IR cameras)
   - Capacitive Hand-on-Detection (HOD) sensors
   - Physiological monitoring (mmWave radar for HRV)

2. **Advanced Deep Learning Models**:
   - CNN-LSTM for temporal drowsiness detection
   - Attention mechanisms for critical feature focus
   - Gated fusion networks for robust sensor integration

3. **Real-World Applications**:
   - SAE Level 3 autonomous driving readiness
   - EURO NCAP compliance
   - Edge deployment on NVIDIA Jetson platforms

---

## 📈 Dataset Information

**Training Data**: 14,584 sequences containing 287,492 frames

| Category | Sequences | Description |
|----------|-----------|-------------|
| Normal Driving | 6,842 | Safe, attentive driving |
| Mobile Phone Use | 3,295 | Texting/calling distractions |
| Distracted | 2,867 | Other attention diversions |
| Drowsy | 1,580 | Fatigue indicators |

---

## 🛠️ System Requirements

- **Python**: 3.7+
- **Processor**: Intel i8 E-2236 or equivalent
- **RAM**: 32 GB recommended
- **GPU**: NVIDIA P2200 or better
- **Frameworks**: TensorFlow 2.8, Scikit-learn 1.0.2

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{biradar2025idss,
  title={A Multi-Modal, Deep Learning Framework for an Integrated Driver Safety System (IDSS)},
  author={Biradar, Amogh and Baria, Maan and Tiwari, Abhinav and Verma, Tanmay and Chauhan, Abhay Singh and Jain, Vipin},
  institution={VIT Bhopal University},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- VIT Bhopal University, School of Computing Science and Engineering
- Dataset providers on Kaggle
- Open-source deep learning community

---

<div align="center">

**⭐ Star this repository if it helps your research!**

*Advancing automotive safety through intelligent driver monitoring*

</div>
