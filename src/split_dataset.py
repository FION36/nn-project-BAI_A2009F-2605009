from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
DATA_PATH = Path("outputs/dataset_info/image_information.csv")
DATASET_PATH = Path(r"C:\Users\fionlol\Documents\NNB_dataset\disaster_dataset")

# Create output folder
OUTPUT_PATH = Path("data_splits")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Read CSV file
df = pd.read_csv(DATA_PATH)

# Remove rows with missing image paths or class labels
df = df.dropna(subset=["image_path", "class"])

# Check images using the actual dataset folder
df = df[
    df["image_path"].apply(
        lambda path: (DATASET_PATH / str(path)).is_file()
    )
]

# Split 70% for training and 30% for temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["class"]
)

# Split the remaining data equally
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"]
)

# Save the three datasets
train_df.to_csv(OUTPUT_PATH / "train.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "validation.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test.csv", index=False)

# Display results
print("========== DATASET SPLITTING ==========")

print("Total images:", len(df))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\n========== CLASS DISTRIBUTION ==========")

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(f"\n{name}:")
    print(data["class"].value_counts().sort_index())

print("\nDataset splitting completed!")