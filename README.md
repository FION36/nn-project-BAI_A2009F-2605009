# Disaster Image Classification Using CNN

## 1. Project Introduction

This project develops a Convolutional Neural Network (CNN) to classify disaster related images into six categories:

1. Fire
2. Flood
3. Landslide
4. Earthquake
5. Smoke
6. Normal

The main objective is to train and evaluate a CNN model using PyTorch and Docker.

The project uses two separate Docker containers:

- **Training Container:** Prepares the environment and trains the CNN model.
- **Testing Container:** Loads the trained model and evaluates its performance using the test dataset.

## 2. Project Structure

```text
NNB_Train/
│
├── README.md
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
├── requirements-test.txt
│
├── src/
│   ├── train.py
│   ├── evaluate_test.py
│   ├── prepare_dataset.py
│   └── model.py
│
├── models/
│   └── best_model.pth
│
├── data/
├── data_splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── outputs/
│   ├── experiments/
│   ├── evaluation/
│   └── dataset_info/
│
└── .venv/
```

The dataset is stored separately from the project folder and mounted into the Docker containers.

## 3. Dataset Information

The dataset contains six image classes.

| Dataset | Number of Images |
|---|---:|
| Training | 6,356 |
| Validation | 1,362 |
| Testing | 1,362 |
| **Total** | **9,080** |

The dataset is divided into training, validation, and testing sets.

## 4. Technologies Used

- Python 3.14
- PyTorch 2.15.0
- Torchvision 0.29.2
- NumPy 2.5.4
- Pandas 3.1.0
- Scikit-learn 1.9.1
- Matplotlib 3.11.2
- Pillow 13.0.0
- Docker

The project uses CPU based PyTorch installation.

## 5. Docker Setup

Make sure Docker Desktop is installed and running.

### Step 1: Build the Training Image

```powershell
docker build -f Dockerfile.train -t nnb-train .
```

### Step 2: Build the Testing Image

```powershell
docker build -f Dockerfile.test -t nnb-test .
```

## 6. Running the Training Container

Run the following command in PowerShell:

```powershell
docker run --rm `
  -v "${PWD}\data_splits:/app/data_splits" `
  -v "${PWD}\outputs:/app/outputs" `
  -v "${PWD}\models:/app/models" `
  -v "C:\Users\fionlol\Documents\NNB_dataset\disaster_dataset:/app/data/disaster_dataset" `
  -e DATASET_ROOT=/app/data/disaster_dataset `
  nnb-train
```

The training container performs the following tasks:

1. Loads the training and validation datasets.
2. Initialises the CNN model.
3. Trains the model for 10 epochs.
4. Evaluates the model using validation data.
5. Saves the best model checkpoint.
6. Saves training history and performance charts.

The trained model is saved in:

```text
models/best_model.pth
```

The experiment results are saved in:

```text
outputs/experiments/docker_baseline/
```

**Note:** Replace the dataset path in the command with the actual dataset location on your computer.

## 7. Running the Testing Container

Run the following command in PowerShell:

```powershell
docker run --rm `
  -v "${PWD}\data_splits:/app/data_splits" `
  -v "${PWD}\outputs:/app/outputs" `
  -v "${PWD}\models:/app/models" `
  -v "C:\Users\fionlol\Documents\NNB_dataset\disaster_dataset:/app/data/disaster_dataset" `
  -e DATASET_ROOT=/app/data/disaster_dataset `
  nnb-test
```

The testing container performs the following tasks:

1. Loads the trained model.
2. Loads the test dataset.
3. Generates predictions.
4. Calculates evaluation metrics.
5. Generates a classification report.
6. Creates a confusion matrix.
7. Saves incorrect prediction examples.

The evaluation results are saved in:

```text
outputs/evaluation/
```

## 8. Model Training Results

The training process completed successfully after 10 epochs.

| Metric | Result |
|---|---:|
| Best Epoch | 10 |
| Training Accuracy | 69.02% |
| Validation Accuracy | 73.94% |
| Best Validation Loss | 0.7745 |

## 9. Final Testing Results

The model was evaluated using 1,362 test images.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 72.47% |
| Macro Precision | 62.19% |
| Macro Recall | 70.57% |
| Macro F1 Score | 62.75% |
| Weighted F1 Score | 73.75% |

### Classification Performance

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Fire | 0.90 | 0.81 | 0.85 |
| Flood | 0.81 | 0.84 | 0.83 |
| Landslide | 0.34 | 0.74 | 0.46 |
| Earthquake | 0.55 | 0.71 | 0.62 |
| Smoke | 0.25 | 0.62 | 0.36 |
| Normal | 0.88 | 0.51 | 0.64 |

The model achieved its strongest performance on Fire and Flood images. Smoke and Landslide classification require further improvement.

## 10. Generated Evaluation Files

The testing container generates the following files:

- `final_test_metrics.csv`
- `target_comparison.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `all_test_predictions.csv`
- `wrong_predictions.csv`
- `wrong_predictions/`

These files are available inside `outputs/evaluation/`.

## 11. Conclusion

This project successfully demonstrates the use of Docker to separate CNN model training and testing into two independent containers.

The training container successfully trained the model and saved its checkpoint. The testing container successfully loaded the trained model and evaluated its performance.

The final test accuracy was 72.47%.

Future improvements may focus on improving classification performance for underrepresented classes, especially Smoke and Landslide.

---

**Project:** Disaster Image Classification Using CNN  
**Framework:** PyTorch  
**Containerisation:** Docker