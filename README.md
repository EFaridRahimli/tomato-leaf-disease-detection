# Tomato Leaf Disease Detection with YOLO

A computer-vision capstone that fine-tunes **YOLOv8** to detect tomato-leaf diseases in field images, benchmarked against **YOLOv5s**. Tomatoes can be hit by 200+ diseases, and manual crop surveys are slow, costly and unreliable. A detector like this automates the inspection.

## Results (test set: 1,051 images)
| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| **YOLOv8 (original fine-tune)** | **0.957** | 0.826 |
| YOLOv8 (refined/tuned) | 0.954 | **0.840** |
| YOLOv5s (benchmark) | 0.924 | 0.688 |

Both YOLOv8 runs clearly beat the YOLOv5s benchmark, especially on the stricter mAP@0.5:0.95 metric. Inference takes ~2–6 ms per image on an RTX 3060 laptop GPU.

| Confusion matrix | Precision-recall curve |
|---|---|
| ![Confusion matrix](model_artifacts/yolov8s_tomato_tuned_TEST_eval/confusion_matrix.png) | ![PR curve](model_artifacts/yolov8s_tomato_tuned_TEST_eval/PR_curve.png) |

## Dataset
[Tomato Leaf Disease Detection – YOLOv8 dataset](https://www.kaggle.com/datasets/vasantharank/tomato-leaf-disease-detection-yolov8-dataset) (Kaggle): YOLO-format annotations, 640×640 images, classes covering healthy leaves and diseases such as Bacterial Spot, Early Blight, Yellow Leaf Curl Virus and Mosaic Virus.

## Approach
1. Problem definition and proposal (`Proposal.pdf`).
2. Fine-tune YOLOv8s on the training split, with augmentation, then a tuned run.
3. Train a YOLOv5s benchmark with the [Ultralytics YOLOv5 repo](https://github.com/ultralytics/yolov5).
4. Evaluate every model on the held-out test set (precision, recall, mAP, per-class metrics).
5. Write up the findings (`project_report.pdf`).

## Repository contents
| Path | Description |
|------|-------------|
| `tomatyolov8.ipynb` | Main notebook: training, evaluation, comparison |
| `model_artifacts/` | Training/test curves, confusion matrices and weights for each run |
| `runs/` | Ultralytics run outputs |
| `yolov5/` | YOLOv5 code used for the benchmark |
| `Proposal.pdf`, `project_report.pdf` | Capstone proposal and final report |
| `requirements.txt` | Python dependencies |

## Run it
```bash
conda create -n tomato python=3.10 && conda activate tomato
pip install -r requirements.txt
jupyter notebook tomatyolov8.ipynb
```
Download the dataset from Kaggle first and update the dataset path (`data.yaml`) in the notebook. Training needs a CUDA GPU.

## Tech stack
Python 3.10 · Ultralytics YOLOv8 · YOLOv5 · PyTorch (CUDA) · pandas · Matplotlib
