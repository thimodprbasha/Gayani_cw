# Gayani CW - AI Healthcare Diagnostics Coursework

## 6COSC020W Coursework: Pneumonia Detection using CNNs

This repository contains the complete coursework submission for AI applications in Healthcare Diagnostics.

## Project Overview

Implementation of a Convolutional Neural Network (CNN) for detecting pneumonia from pediatric chest X-ray images.

## Contents

| File | Description |
|------|-------------|
| `AI_Healthcare_Coursework.md` | Complete written coursework (Parts A-E) |
| `Pneumonia_Detection_CNN.ipynb` | Jupyter notebook with CNN implementation |
| `Video_Script.md` | 3-minute demo video script |
| `Assignment_Plan.md` | Original assignment breakdown |

## Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow jupyter
```

3. Download dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract to `./chest_xray`

4. Run Jupyter notebook:
```bash
jupyter notebook Pneumonia_Detection_CNN.ipynb
```

## Model Performance

- **Accuracy**: ~90%
- **Recall**: ~95%
- **Precision**: ~85%
- **AUC**: ~0.92

## Technology Stack

- Python 3.x
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## Author

Gayani - 6COSC020W Coursework Submission
