import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

from model import DisasterCNN

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
    type=str,
    required=True
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.001
)

parser.add_argument(
    "--dropout",
    type=float,
    default=0.3
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10
)

args = parser.parse_args()


#REPRODUCIBILITY

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Import prepared datasets after setting seed
from prepare_dataset import (
    train_loader,
    val_loader,
    weights,
    CLASSES
)


#DEVICE AND OUTPUT FOLDER
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

OUTPUT_DIR = Path("outputs/experiments") / args.name

MODEL_DIR = OUTPUT_DIR / "models"
SHARED_MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SHARED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Experiment:", args.name)
print("Learning rate:", args.lr)
print("Dropout:", args.dropout)
print("Epochs:", args.epochs)


#CNN MODEL
model = DisasterCNN(
    num_classes=6,
    dropout_p=args.dropout
)

model = model.to(DEVICE)

#LOSS FUNCTION
weights = weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=weights
)

#OPTIMISER
optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr
)


#VALIDATION FUNCTION
def validate_model(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs, dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


#TRAINING
history = []

best_val_loss = float("inf")
best_epoch = 0

for epoch in range(args.epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Calculate training statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs, dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    # Training metrics
    train_loss = running_loss / total
    train_accuracy = correct / total


    # Validation metrics
    val_loss, val_accuracy = validate_model(
        model,
        val_loader
    )


    # Record results
    history.append({

        "Epoch": epoch + 1,

        "Train Loss": train_loss,

        "Train Accuracy": train_accuracy,

        "Validation Loss": val_loss,

        "Validation Accuracy": val_accuracy

    })


    # Display progress
    print(
        f"Epoch [{epoch + 1}/{args.epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # Save best model based on validation loss
if val_loss < best_val_loss:

    best_val_loss = val_loss

    best_epoch = epoch + 1

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASSES,
        "input_size": 224,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "epoch": best_epoch,
        "best_val_loss": best_val_loss
    }

    # Save inside the experiment folder
    torch.save(
        checkpoint,
        MODEL_DIR / "best_model.pth"
    )

    # Save a shared copy for the testing container
    torch.save(
        checkpoint,
        SHARED_MODEL_DIR / "best_model.pth"
    )

    print("Best model saved!")
    print("Shared model copy saved!")


#TRAINING HISTORY

history_df = pd.DataFrame(history)

history_df.to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)


#TRAINING LOSS
plt.figure(figsize=(8, 5))

plt.plot(
    history_df["Epoch"],
    history_df["Train Loss"],
    label="Training Loss"
)

plt.plot(
    history_df["Epoch"],
    history_df["Validation Loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(args.name + " Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "loss_curve.png"
)

plt.close()


#TRAINING ACCURACY
plt.figure(figsize=(8, 5))

plt.plot(
    history_df["Epoch"],
    history_df["Train Accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_df["Epoch"],
    history_df["Validation Accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(args.name + " Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_DIR / "accuracy_curve.png"
)

plt.close()


#EXPERIMENT SUMMARY
summary = {

    "Experiment": args.name,

    "Learning Rate": args.lr,

    "Dropout": args.dropout,

    "Epochs": args.epochs,

    "Best Epoch": best_epoch,

    "Best Validation Loss": best_val_loss,

    "Best Validation Accuracy": history_df.loc[
        best_epoch - 1,
        "Validation Accuracy"
    ]

}

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False
)

print("\n=================================")
print("TRAINING COMPLETED")
print("=================================")

print("Experiment:", args.name)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)

print("Results saved in:", OUTPUT_DIR)
