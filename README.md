# Helmet Detection Using YOLO

A computer vision project for detecting whether a person is wearing a helmet using YOLO-based object detection models.

This project trains and evaluates three lightweight YOLO models:

- YOLOv7-tiny
- YOLOv8n
- YOLO11n

All three models are trained on the same helmet detection dataset and evaluated on a separate held-out test set using Precision, Recall, mAP@0.5, and mAP@0.5:0.95.

---

## Project Overview

The goal of this project is to build an object detection system that identifies two classes:

- `helmet`
- `no_helmet`

The project compares YOLOv7-tiny, YOLOv8n, and YOLO11n using the same dataset and comparable training conditions.

The project covers:

- Dataset preparation
- YOLO annotation visualization
- Object detection model training
- Model evaluation
- Held-out test-set evaluation
- Prediction visualization
- Comparison of multiple YOLO model versions

---

## Classes

| Class ID | Class Name |
|---------:|------------|
| 0 | helmet |
| 1 | no_helmet |

---

## Dataset

The dataset contains **1,376 images** divided into training, validation, and test sets.

| Split | Images |
|-------|-------:|
| Train | 1,185 |
| Validation | 127 |
| Test | 64 |
| **Total** | **1,376** |

The dataset is organized using the YOLO object detection format.

dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

The annotations use YOLO bounding-box format.

### Dataset Source

The dataset was obtained from Roboflow:

**Bike Helmet Detection - v1**

Source:

[https://universe.roboflow.com/yolo-v11-mmc6o/bike-helmet-detection-2vdjo-agpe6](https://universe.roboflow.com/yolo-v11-mmc6o/bike-helmet-detection-2vdjo-agpe6)

Dataset attribution and export information are preserved in:

```text
dataset/README.dataset.txt
dataset/README.roboflow.txt
```

---

## Dataset Configuration

The project uses the following dataset configuration:

```yaml
path: ./dataset

train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: no_helmet
```

The same dataset split was used for all three model experiments.

---

## Models

Three lightweight YOLO models were trained and evaluated.

| Model       | Implementation                 |
| ----------- | ------------------------------ |
| YOLOv7-tiny | Original YOLOv7 implementation |
| YOLOv8n     | Ultralytics                    |
| YOLO11n     | Ultralytics                    |

YOLOv7-tiny was trained using the original YOLOv7 implementation and its training scripts.

YOLOv8n and YOLO11n were trained using the Ultralytics framework.

---

## Training Configuration

The main experiments used the following configuration:

| Parameter  | Value     |
| ---------- | --------- |
| Image size | 640 × 640 |
| Epochs     | 30        |
| Batch size | 8         |
| Device     | CPU       |
| Workers    | 2         |

The experiments were performed on a CPU-only system.

The same dataset, image size, number of epochs, and batch size were used for the three model experiments.

---

## Test Results

The models were evaluated on the **64-image held-out test set containing 187 annotated objects**.

### Overall Performance

| Model       | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------- | --------: | -----: | ------: | -----------: |
| YOLOv7-tiny |     0.777 |  0.793 |   0.806 |        0.442 |
| YOLOv8n     |     0.883 |  0.725 |   0.813 |        0.441 |
| YOLO11n     |     0.811 |  0.769 |   0.833 |        0.440 |

### Helmet Class

| Model       | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------- | --------: | -----: | ------: | -----------: |
| YOLOv7-tiny |     0.861 |  0.903 |   0.942 |        0.566 |
| YOLOv8n     |     0.939 |  0.847 |   0.943 |        0.545 |
| YOLO11n     |     0.917 |  0.894 |   0.957 |        0.543 |

### No-Helmet Class

| Model       | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------- | --------: | -----: | ------: | -----------: |
| YOLOv7-tiny |     0.693 |  0.683 |   0.671 |        0.318 |
| YOLOv8n     |     0.827 |  0.603 |   0.684 |        0.337 |
| YOLO11n     |     0.705 |  0.645 |   0.708 |        0.337 |

The results show different precision and recall characteristics across the three models.

The `no_helmet` class has lower recall than the `helmet` class for all three models.

The overall mAP@0.5:0.95 values are close across the three experiments.

---

## Training Time

| Model       | Training Time |
| ----------- | ------------: |
| YOLOv7-tiny |  18.286 hours |
| YOLOv8n     |   4.087 hours |
| YOLO11n     |   4.317 hours |

Training was performed on a CPU-only system.

These times are specific to the hardware and software configuration used for this project and should not be interpreted as general benchmark results.

---

## Model Weights

The trained model weights are organized as follows:

```text
models/
├── yolov7-tiny/
│   └── best.pt
├── yolov8n/
│   └── best.pt
└── yolo11n/
    └── best.pt
```

The weights included in the project are the best checkpoints obtained from the respective training runs.

---

## Project Structure

```text
helmet-detection/
├── README.md
├── data.yaml
├── requirements.txt
├── requirements-yolov7.txt
├── .gitignore
│
├── dataset/
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── yolov7-tiny/
│   │   └── best.pt
│   ├── yolov8n/
│   │   └── best.pt
│   └── yolo11n/
│       └── best.pt
│
├── results/
│   ├── metrics/
│   │   ├── yolov7-tiny/
│   │   ├── yolov8n/
│   │   └── yolo11n/
│   │
│   └── training/
│       ├── yolov7-tiny/
│       ├── yolov8n/
│       └── yolo11n/
│
├── notebooks/
│   └── dataset_visualization.ipynb
│
└── src/
    └── predict.py
```

The `results/predictions/` directory is generated locally during inference and is excluded from Git using `.gitignore`.

The original YOLOv7 source repository used during training is not included in the final portfolio project structure.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sowjanya5606/helmet-detection.git
cd helmet-detection
```

### 2. Create a virtual environment

```bash
python -m venv helmet
```

### 3. Activate the environment on Windows

```bat
helmet\Scripts\activate
```

### 4. Install the main dependencies

```bash
pip install -r requirements.txt
```

The main requirements are used for YOLOv8n and YOLO11n inference and project utilities.

### YOLOv7 Training Environment

YOLOv7 was trained using the original YOLOv7 implementation with a separate compatible environment.

The corresponding dependency file is provided as:

```text
requirements-yolov7.txt
```

Install it in a separate environment if reproducing the YOLOv7 training setup:

```bash
pip install -r requirements-yolov7.txt
```

The original YOLOv7 source repository is not included in this portfolio repository.

---

## Inference

The project includes a prediction script for the Ultralytics-based models:

```text
src/predict.py
```

The script currently supports:

* YOLOv8n
* YOLO11n

### YOLO11n

Run inference on the test images:

```bash
python src/predict.py --model yolo11n --source dataset/images/test
```

### YOLOv8n

```bash
python src/predict.py --model yolov8n --source dataset/images/test
```

### Custom Image

```bash
python src/predict.py --model yolo11n --source path/to/image.jpg
```

### Custom Folder

```bash
python src/predict.py --model yolo11n --source path/to/folder
```

### Custom Confidence Threshold

```bash
python src/predict.py --model yolo11n --source dataset/images/test --conf 0.40
```

### Custom Image Size

```bash
python src/predict.py --model yolo11n --source dataset/images/test --imgsz 640
```

### Output

Prediction results generated by the script are saved locally under:

```text
results/predictions/inference/
```

For example:

```text
results/predictions/inference/
└── yolo11n/
```

Prediction outputs are generated locally by `src/predict.py` and are excluded from Git using `.gitignore`.

They are not stored in the GitHub repository.

### YOLOv7-tiny

YOLOv7-tiny was trained and evaluated using the original YOLOv7 implementation.

Its trained weights and evaluation artifacts are included in this project.

Prediction images generated during inference are stored locally and excluded from Git.

The original YOLOv7 source code is not included in the portfolio repository. New YOLOv7 inference therefore requires the original YOLOv7 implementation to be installed separately.

---

## Dataset Visualization

The project includes a Jupyter notebook for inspecting the dataset annotations:

```text
notebooks/dataset_visualization.ipynb
```

The notebook:

* Loads sample training images
* Reads YOLO annotations
* Draws bounding boxes
* Displays class labels
* Helps visually inspect the dataset annotations

---

## Evaluation Outputs

Evaluation artifacts committed to the repository are stored under:

```text
results/
├── metrics/
└── training/
```

### Metrics

The metrics directories contain evaluation outputs such as:

* Precision curves
* Recall curves
* F1 curves
* Precision-Recall curves
* Confusion matrices
* Validation batch visualizations

### Training

The training directories contain artifacts such as:

* Training curves
* Training results
* Confusion matrices
* YOLOv7 training configuration and results

### Predictions

Prediction images are generated locally during inference and are excluded from Git using `.gitignore`.

They are saved locally under:

```text
results/predictions/
```

---

## Evaluation Metrics

### Precision

Precision measures the proportion of predicted detections that are correct.

```text
Precision = TP / (TP + FP)
```

where:

* TP = True Positives
* FP = False Positives

### Recall

Recall measures the proportion of actual objects that were successfully detected.

```text
Recall = TP / (TP + FN)
```

where:

* TP = True Positives
* FN = False Negatives

### mAP@0.5

Mean Average Precision calculated using an IoU threshold of 0.5.

### mAP@0.5:0.95

Mean Average Precision averaged across IoU thresholds from 0.5 through 0.95.

---

## Key Observations

* All three models successfully learned the two-class helmet detection task.
* The `helmet` class achieved higher recall than the `no_helmet` class for all three models.
* The `no_helmet` class was more challenging to detect consistently.
* The models showed different precision and recall characteristics.
* Overall mAP@0.5:0.95 values were close across the three experiments.
* YOLOv7-tiny required substantially more training time in this CPU-based setup than YOLOv8n and YOLO11n.
* The reported metrics are based on the same held-out test set for all three models.

---

## Limitations

* Training was performed on a CPU-only system.
* The dataset contains more `helmet` annotations than `no_helmet` annotations.
* `no_helmet` detection has lower recall than `helmet` detection across the evaluated models.
* The experiments used 30 training epochs.
* Results may change with different datasets, hyperparameters, augmentation strategies, or training durations.
* The test set contains only 64 images, so the reported metrics should be interpreted in the context of this dataset.
* The project currently focuses on object detection and does not include a complete production deployment pipeline.
* Real-world performance may differ from the reported test-set results.

---

## Future Improvements

Possible extensions include:

* Increase the size and diversity of the dataset.
* Improve class balance.
* Perform hyperparameter tuning.
* Experiment with additional data augmentation.
* Train for more epochs with appropriate early stopping.
* Test larger YOLO model variants.
* Add real-time webcam detection.
* Add video inference.
* Export models to ONNX or other deployment formats.
* Optimize models for edge devices.
* Build a simple web or desktop interface for helmet detection.
* Evaluate the models on additional real-world datasets.

---

## Technologies Used

* Python
* PyTorch
* Ultralytics
* YOLOv7
* YOLOv8
* YOLO11
* OpenCV
* NumPy
* pandas
* Matplotlib
* Pillow
* PyYAML
* Jupyter Notebook

---

## Author

**Sowjanya**

B.Tech, 2024
