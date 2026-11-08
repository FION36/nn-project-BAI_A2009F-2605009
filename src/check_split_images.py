
from pathlib import Path
import pandas as pd
from PIL import Image, UnidentifiedImageError

# Dataset split folder
DATA_PATH = Path("data_splits")

# Output report
OUTPUT_PATH = Path("outputs/dataset_info")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

split_files = [
    "train.csv",
    "validation.csv",
    "test.csv"
]

corrupted_images = []


def check_image(image_path):

    try:
        # First check image structure
        with Image.open(image_path) as img:
            img.verify()

        # Reopen and fully load the image
        with Image.open(image_path) as img:
            img.convert("RGB").load()

        return True

    except (OSError, ValueError, UnidentifiedImageError) as error:

        return str(error)


for filename in split_files:

    csv_path = DATA_PATH / filename

    df = pd.read_csv(csv_path)

    valid_rows = []

    print(f"\nChecking {filename}...")

    for _, row in df.iterrows():

        image_path = str(row["image_path"])

        result = check_image(image_path)

        if result is True:

            valid_rows.append(row)

        else:

            print("Invalid image:", image_path)

            corrupted_images.append({
                "image_path": image_path,
                "class": row["class"],
                "split": filename,
                "error": result
            })

    # Keep only valid images
    clean_df = pd.DataFrame(
        valid_rows,
        columns=df.columns
    )

    clean_df.to_csv(csv_path, index=False)

    print("Original images:", len(df))
    print("Valid images:", len(clean_df))
    print("Removed images:", len(df) - len(clean_df))


# Save corrupted image report
corrupted_df = pd.DataFrame(corrupted_images)

corrupted_df.to_csv(
    OUTPUT_PATH / "corrupted_images.csv",
    index=False
)

print("\n================================")
print("IMAGE CHECK COMPLETED")
print("================================")

print("Total corrupted images:", len(corrupted_images))

print("Report saved to:")
print(OUTPUT_PATH / "corrupted_images.csv")
