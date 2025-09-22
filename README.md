# Real-Life Face Occlusion Using Deep Learning & Computer Vision

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.14-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

**Real-Life Face Occlusion Using DL & CV** is a major project focused on detecting occlusions in human faces such as **masks, hats, and glasses**. The project leverages **Deep Learning** and **Computer Vision** to improve face recognition in real-world scenarios.

---

## Features

* Detect faces from images or videos in real-time.
* Identify and classify occlusions: masks, hats, glasses, etc.
* Handle multiple faces in a single frame.
* Improve downstream face recognition under occlusions.

---

## Repository Structure

```
Real-Life-Face-Occlusion-Using-DL-CV/
│
├── data/                  # Dataset with faces and occlusion images
├── models/                # Trained deep learning models
├── notebooks/             # Jupyter notebooks for experimentation & training
├── scripts/               # Python scripts for training and inference
├── app.py                 # Real-time Face Occlusion Detection App
├── face-occlusiontrain.ipynb  # Model training notebook
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Dataset

* Original: CelebHQ & custom datasets.
* Synthetic occlusions added: masks, hats, glasses.
* Image resolution: **218x178** pixels.
* Annotated with occlusion type and location.

---

## Technology Stack

* **Programming Language:** Python 3.x
* **Deep Learning Framework:** PyTorch / TensorFlow
* **Computer Vision Libraries:** OpenCV, Pillow
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ashishkumar43/Real-Life-Face-Occlusion-Using-DL-CV.git
cd Real-Life-Face-Occlusion-Using-DL-CV
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python scripts/train.py --data_path ./data --model_path ./models
```

* Set hyperparameters like learning rate, batch size, and epochs inside the script.

### Inference / Testing

```bash
python app.py --image_path ./data/test_image.jpg --model_path ./models/face_occlusion_model.pth
```

* Detects faces and highlights occluded regions in images or real-time video.

---

## Applications

* Face recognition in public spaces (mask compliance).
* Security and surveillance.
* AR/VR filters for virtual masks or glasses.
* Healthcare monitoring.

---

## Future Enhancements

* Real-time high FPS video stream support.
* Additional occlusion types (scarves, hands, objects).
* Web or mobile deployment.
* Integration with advanced face recognition pipelines.

---

## References

* [CelebHQ Dataset](https://github.com/some-dataset-link)
* [OpenCV](https://opencv.org/)
* [PyTorch](https://pytorch.org/)

---

## Author

**Harsh Kumar Jha**
Final Year B.Tech | Data Science & Computer Vision Enthusiast

---
