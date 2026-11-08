import torch
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from model import DisasterCNN
from prepare_dataset import (
    train_dataset,
    test_dataset,
    CLASSES,
    DATASET_ROOT
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path(
    "models/best_model.pth"
)

OUTPUT_DIR = Path("outputs/evaluation")
WRONG_DIR = OUTPUT_DIR / "wrong_predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WRONG_DIR.mkdir(parents=True, exist_ok=True)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Device:", DEVICE)
print("Test images:", len(test_dataset))


#LOAD BASELINE CHECKPOINT
print("\nLoading selected Baseline model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Check that the checkpoint uses the expected class order.
if checkpoint["class_names"] != CLASSES:
    raise ValueError(
        "Class order in checkpoint does not match dataset class order."
    )

model = DisasterCNN(
    num_classes=len(CLASSES),
    dropout_p=checkpoint["dropout"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Baseline model loaded successfully.")
print("Checkpoint epoch:", checkpoint["epoch"])


#EVALUATE TEST SET ONCE
print("\nEvaluating the test dataset...")

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

print("Test evaluation completed.")


#CALCULATE CLASSIFICATION METRICS
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall:    {recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")


#CLASSIFICATION REPORT
report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as file:
    file.write(report)


#CONFUSION MATRIX
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASSES)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45,
    colorbar=False
)

plt.title("Baseline Model Confusion Matrix")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

print("Confusion matrix saved.")


#MAJORITY CLASS BASELINE
# Find the most common class in the training dataset.
train_class_counts = train_dataset.data["class"].value_counts()

majority_class = train_class_counts.idxmax()

majority_class_index = CLASSES.index(majority_class)

# Predict the training majority class for every test image.
baseline_predictions = [
    majority_class_index
] * len(all_labels)

majority_accuracy = accuracy_score(
    all_labels,
    baseline_predictions
)

print("\n========== MAJORITY CLASS BASELINE ==========")
print("Most common training class:", majority_class)
print(f"Majority baseline accuracy: {majority_accuracy:.4f}")
print(f"CNN test accuracy:           {accuracy:.4f}")


#COMPARE WITH TASK 1 TARGETS
comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],

    "Task 1 Target": [
        0.80,
        0.75,
        0.75,
        0.75
    ],

    "Test Result": [
        accuracy,
        precision,
        recall,
        macro_f1
    ]
})

comparison["Target Achieved"] = (
    comparison["Test Result"] >=
    comparison["Task 1 Target"]
)

comparison.to_csv(
    OUTPUT_DIR / "target_comparison.csv",
    index=False
)

print("\n========== TASK 1 TARGET COMPARISON ==========")
print(comparison.to_string(index=False))


#SAVE ALL TEST PREDICTIONS
prediction_data = []

for index, (actual, predicted) in enumerate(
    zip(all_labels, all_predictions)
):

    prediction_data.append({

        "Image Path": test_dataset.data.iloc[index]["image_path"],

        "Actual Class": CLASSES[actual],

        "Predicted Class": CLASSES[predicted],

        "Correct": actual == predicted

    })

predictions_df = pd.DataFrame(prediction_data)

predictions_df.to_csv(
    OUTPUT_DIR / "all_test_predictions.csv",
    index=False
)


#SAVE WRONG PREDICTION EXAMPLES
wrong_indices = [
    index
    for index, (actual, predicted)
    in enumerate(zip(all_labels, all_predictions))
    if actual != predicted
]

print("\n========== WRONG PREDICTIONS ==========")
print("Total incorrect predictions:", len(wrong_indices))

wrong_examples = []

# Save up to 10 incorrect images for report analysis.
for number, index in enumerate(
    wrong_indices[:10],
    start=1
):

    relative_path = test_dataset.data.iloc[index]["image_path"]

    image_path = DATASET_ROOT / str(relative_path)

    actual_class = CLASSES[all_labels[index]]

    predicted_class = CLASSES[all_predictions[index]]

    image = Image.open(image_path).convert("RGB")

    save_path = WRONG_DIR / f"wrong_{number}.jpg"

    image.save(save_path)

    wrong_examples.append({

        "Example": number,

        "Image Path": relative_path,

        "Actual Class": actual_class,

        "Predicted Class": predicted_class,

        "Saved Image": str(save_path)

    })

    print(
        f"{number}. Actual: {actual_class} | "
        f"Predicted: {predicted_class}"
    )


wrong_df = pd.DataFrame(wrong_examples)

wrong_df.to_csv(
    OUTPUT_DIR / "wrong_predictions.csv",
    index=False
)


#SAVE FINAL METRICS
metrics = pd.DataFrame([{

    "Model": "Baseline",

    "Accuracy": accuracy,

    "Macro Precision": precision,

    "Macro Recall": recall,

    "Macro F1": macro_f1,

    "Weighted F1": weighted_f1,

    "Majority Class": majority_class,

    "Majority Baseline Accuracy": majority_accuracy

}])

metrics.to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False
)


print("\n===================================")
print("TEST EVALUATION COMPLETED")
print("===================================")

print("Results:", OUTPUT_DIR)

print("\nGenerated files:")
print("1. final_test_metrics.csv")
print("2. target_comparison.csv")
print("3. classification_report.txt")
print("4. confusion_matrix.png")
print("5. all_test_predictions.csv")
print("6. wrong_predictions.csv")
print("7. wrong_predictions folder")
