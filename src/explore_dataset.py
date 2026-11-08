
from pathlib import Path
from PIL import Image
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load inspection information
DATA_PATH = Path("outputs/dataset_info/image_information.csv")

OUTPUT_PATH = Path("outputs/charts")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

CLASSES = [
    "Fire",
    "Flood",
    "Landslide",
    "Earthquake",
    "Smoke",
    "Normal"
]

# V1: CLASS DISTRIBUTION
class_counts = (
    df["class"]
    .value_counts()
    .reindex(CLASSES, fill_value=0)
)

plt.figure(figsize=(10, 6))

class_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Disaster Dataset Class Distribution")
plt.xlabel("Disaster Category")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "class_distribution.png",
    dpi=300
)

plt.show()

print("Class distribution chart saved.")


# V2: RGB COLOUR DISTRIBUTION
fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 8)
)

random.seed(42)

for index, class_name in enumerate(CLASSES):

    class_images = df[
        df["class"] == class_name
    ]["image_path"].tolist()

    if not class_images:
        continue

    # Randomly select up to 50 images
    selected_images = random.sample(
        class_images,
        min(50, len(class_images))
    )

    # Store RGB histograms
    rgb_histograms = {
        "Red": [],
        "Green": [],
        "Blue": []
    }

    for selected_image in selected_images:

        with Image.open(selected_image) as img:
            image = np.array(
                img.convert("RGB").resize((128, 128))
            )

        # Calculate RGB histograms
        for channel, color in enumerate(
            ["Red", "Green", "Blue"]
        ):

            histogram, _ = np.histogram(
                image[:, :, channel],
                bins=256,
                range=(0, 256)
            )

            # Convert to percentage distribution
            histogram = histogram / histogram.sum()

            rgb_histograms[color].append(histogram)

    row = index // 3
    col = index % 3

    for color, line_color in zip(
        ["Red", "Green", "Blue"],
        ["red", "green", "blue"]
    ):

        # Calculate average histogram
        average_histogram = np.mean(
            rgb_histograms[color],
            axis=0
        )

        axes[row, col].plot(
            average_histogram,
            color=line_color,
            label=color
        )

    axes[row, col].set_title(class_name)
    axes[row, col].set_xlabel("Pixel Intensity")
    axes[row, col].set_ylabel("Average Pixel Proportion")
    axes[row, col].legend()
    axes[row, col].grid(alpha=0.2)

plt.suptitle("RGB Colour Distribution by Disaster Category")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "rgb_distribution.png",
    dpi=300
)

plt.show()

print("RGB colour distribution chart saved.")


# V3: IMAGE DIMENSIONS

plt.figure(figsize=(10, 6))

plt.scatter(
    df["width"],
    df["height"],
    alpha=0.5
)

plt.title("Original Image Dimensions")
plt.xlabel("Image Width (pixels)")
plt.ylabel("Image Height (pixels)")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "image_dimensions.png",
    dpi=300
)

plt.show()

print("Image dimensions chart saved.")
