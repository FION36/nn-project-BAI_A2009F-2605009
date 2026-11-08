import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from model import DisasterCNN
from prepare_dataset import val_loader, CLASSES

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model checkpoint paths
MODEL_PATHS = {
    "Baseline": Path(
        "outputs/experiments/baseline/models/best_model.pth"
    ),
    "Experiment 1": Path(
        "outputs/experiments/experiment_lr/models/best_model.pth"
    ),
    "Experiment 2": Path(
        "outputs/experiments/experiment_dropout/models/best_model.pth"
    )
}

results = []

# Evaluate each model using validation data
for model_name, model_path in MODEL_PATHS.items():

    print(f"\nEvaluating {model_name}...")

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = DisasterCNN(
        num_classes=6,
        dropout_p=checkpoint["dropout"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels, all_predictions
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

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Macro F1": f1
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")


# Compare all models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

# Save results
output_path = Path(
    "outputs/experiments/validation_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n========== VALIDATION COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nBest model based on Macro F1:")
print(results_df.iloc[0]["Model"])

print("\nResults saved to:", output_path)