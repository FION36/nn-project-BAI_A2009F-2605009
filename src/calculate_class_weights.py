from pathlib import Path
import pandas as pd
import json

# Load training dataset only
DATA_PATH = Path("data_splits/train.csv")

OUTPUT_PATH = Path("outputs/dataset_info")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Read training data
df = pd.read_csv(DATA_PATH)

# Define the six classes
CLASSES = [
    "Fire",
    "Flood",
    "Landslide",
    "Earthquake",
    "Smoke",
    "Normal"
]

# Count images in each class
class_counts = df["class"].value_counts().reindex(
    CLASSES,
    fill_value=0
)

# Check that every class has training images
if (class_counts == 0).any():
    raise ValueError("At least one class has no training images.")

# Calculate class weights
total_images = len(df)
number_of_classes = len(CLASSES)

class_weights = {}

for class_name in CLASSES:
    weight = total_images / (
        number_of_classes * class_counts[class_name]
    )

    class_weights[class_name] = float(weight)

# Save class weights
with open(OUTPUT_PATH / "class_weights.json", "w") as file:
    json.dump(class_weights, file, indent=4)

# Display results
print("========== CLASS WEIGHTS ==========")

for class_name, weight in class_weights.items():
    print(f"{class_name}: {weight:.4f}")

print("\nClass weights saved successfully!")