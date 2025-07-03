# Binary Image Classification (Cat vs. Rabbit)

A Google Colab notebook for training, validating, and evaluating a binary image classifier (cats vs. rabbits) using a YOLOv5‑based model in PyTorch. This project demonstrates data integrity checks, exploratory data analysis, model training via the Ultralytics YOLOv5 classification script, and post‑training evaluation.
The training images can be downloaded from:

https://www.kaggle.com/datasets/muniryadi/cat-vs-rabbit


## Overview

This notebook walks through:

1. **Data Preparation**  
   - Mounting Google Drive to access image folders.  
   - Verifying file integrity (valid image formats).  
   - Computing class statistics and visualizing the dataset distribution.

2. **Model Training**  
   - Invoking the YOLOv5 classification training script (`classify/train.py`) via Colab shell.  
   - Customizing hyperparameters (model variant, number of epochs, image size).

3. **Evaluation**  
   - Loading the best‐performing checkpoint (`best.pt`).  
   - Validating on separate “cat” and “rabbit” validation sets.  
   - Reporting per‐class accuracy and plotting the results.

---

## Quick Start

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/binary-image-classification.git
   cd binary-image-classification



2. **Install the missing dependencies**

```bash
pip install torch torchvision opencv-python matplotlib tqdm
```

Within the notebook you can customize:

!python classify/train.py \
  --model yolov5s-cls.pt \
  --data /content/drive/MyDrive/ml/ \
  --epochs 5 \
  --img 224 \
  --cache


Where,

--model: Pretrained YOLOv5 classification weights

--data: Path to your Drive folder containing train/ and val/ subdirectories

--epochs: Number of training epochs

--img: Input image resolution

--cache: Cache images in RAM for faster training

**Project Structure**

binary-image-classification/
├── binary_image_classification.ipynb   # Google Colab notebook
├── requirements.txt                    # Python dependencies (optional)
├── classify/                           # YOLOv5 classification code
│   └── train.py                        # Training script entry point
└── README.md                           # This file

