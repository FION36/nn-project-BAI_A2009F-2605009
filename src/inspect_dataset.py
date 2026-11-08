
from pathlib import Path
from PIL import Image, ImageFile
import pandas as pd

# CONFIGURATION
DATASET_PATH = Path(
    r"C:\Users\fionlol\Documents\NNB_dataset\disaster_dataset"
)

CLASSES = [
    "Fire",
    "Flood",
    "Landslide",
    "Earthquake",
    "Smoke",
    "Normal"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

OUTPUT_PATH = Path("outputs/dataset_info")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Do not allow incomplete images to load silently
ImageFile.LOAD_TRUNCATED_IMAGES = False


# CHECK DATASET LOCATION
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


print("\n========== DATASET INSPECTION ==========\n")

# INSPECT EACH CLASS
image_records = []
corrupted_images = []

for class_name in CLASSES:

    class_path = DATASET_PATH / class_name

    if not class_path.is_dir():
        print(f"WARNING: Missing folder: {class_name}")
        continue

    # Include images in nested subfolders
    images = [
        file for file in class_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    readable_count = 0

    for image_path in images:

        try:
            # First check image structure
            with Image.open(image_path) as img:
                img.verify()

            # Reopen and fully load image pixels
            with Image.open(image_path) as img:

                width = img.width
                height = img.height
                image_format = img.format

                # Convert to RGB and force complete loading
                rgb_image = img.convert("RGB")
                rgb_image.load()

                rgb_image.close()

            # Record only fully readable images
            image_records.append({
                "image_path": image_path.relative_to(DATASET_PATH).as_posix(),
                "class": class_name,
                "width": width,
                "height": height,
                "format": image_format
            })

            readable_count += 1

        except Exception as error:

            corrupted_images.append({
                "image_path": image_path.relative_to(DATASET_PATH).as_posix(),
                "class": class_name,
                "error": str(error)
            })

    print(f"{class_name}:")
    print(f"  Total images: {len(images)}")
    print(f"  Readable images: {readable_count}")
    print(f"  Corrupted images: {len(images) - readable_count}")


# SAVE RESULTS
df = pd.DataFrame(image_records)

corrupted_df = pd.DataFrame(corrupted_images)

df.to_csv(
    OUTPUT_PATH / "image_information.csv",
    index=False
)

corrupted_df.to_csv(
    OUTPUT_PATH / "corrupted_images.csv",
    index=False
)


# SUMMARY
print("\n========== SUMMARY ==========")

print("Total images checked:",
      len(image_records) + len(corrupted_images))

print("Total readable images:", len(df))

print("Total corrupted images:", len(corrupted_df))

print("\nReadable images by class:")

print(
    df["class"].value_counts().reindex(
        CLASSES,
        fill_value=0
    )
)

print("\nImage formats:")
print(df["format"].value_counts())

print("\nInspection completed successfully!")

print("Reports saved in:", OUTPUT_PATH.resolve())
