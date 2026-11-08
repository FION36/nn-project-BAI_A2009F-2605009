from pathlib import Path
import pandas as pd
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define paths
DATA_PATH = Path("data_splits")
WEIGHT_PATH = Path("outputs/dataset_info/class_weights.json")
OUTPUT_PATH = Path("outputs/dataset_info")

# Dataset location
DATASET_ROOT = Path(
    os.environ.get(
        "DATASET_ROOT",
        r"C:\Users\fionlol\Documents\NNB_dataset\disaster_dataset"
    )
)

# Create output folder if it does not exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Define classes and their numerical labels
CLASSES = [
    "Fire",
    "Flood",
    "Landslide",
    "Earthquake",
    "Smoke",
    "Normal"
]

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(CLASSES)
}

# Save class mapping
with open(OUTPUT_PATH / "class_mapping.json", "w") as file:
    json.dump(class_to_idx, file, indent=4)

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    )
])

# Validation and testing transformations
evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Create custom PyTorch Dataset
class DisasterDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        relative_path = self.data.iloc[index]["image_path"]
        class_name = self.data.iloc[index]["class"]

        # Combine dataset root with relative image path
        image_path = DATASET_ROOT / str(relative_path)

        # Load image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Convert class name into numerical label
        label = class_to_idx[class_name]

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Create three datasets
train_dataset = DisasterDataset(
    DATA_PATH / "train.csv",
    transform=train_transform
)

val_dataset = DisasterDataset(
    DATA_PATH / "validation.csv",
    transform=evaluation_transform
)

test_dataset = DisasterDataset(
    DATA_PATH / "test.csv",
    transform=evaluation_transform
)


# Create DataLoaders
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load class weights
with open(WEIGHT_PATH, "r") as file:
    class_weights = json.load(file)

weights = torch.tensor(
    [class_weights[name] for name in CLASSES],
    dtype=torch.float32
)

# Display dataset information
print("========== DATASET INFORMATION ==========")

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

print("\nClass mapping:")
print(class_to_idx)

print("\nClass weights:")
print(weights)

# Check one batch
images, labels = next(iter(train_loader))

print("\n========== BATCH CHECK ==========")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Minimum pixel value:", images.min().item())
print("Maximum pixel value:", images.max().item())

print("\nDataset preparation completed!")