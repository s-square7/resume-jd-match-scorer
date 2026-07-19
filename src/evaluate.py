"""
Evaluation utilities: classification report, macro-F1, confusion matrix.

Author : Shuvam Saren (25MA60R16), IIT Kharagpur

Because the Weak class is easy and Medium vs Strong is hard, accuracy alone is
misleading. We foreground MACRO-F1 (equal weight to every class) and always
print the full per-class report and confusion matrix so the Medium/Strong
confusion is visible.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score


LABEL_NAMES = {0: "Weak Match", 1: "Medium Match", 2: "Strong Match"}


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_NAMES[i] for i in range(3)],
        yticklabels=[LABEL_NAMES[i] for i in range(3)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Resume-JD Match Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return cm


def build_classification_report(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        target_names=[LABEL_NAMES[i] for i in range(3)],
        digits=4,
    )
    header = (
        "Macro-F1 (primary metric, equal weight per class): "
        f"{macro_f1:.4f}\n"
        "Read the Medium/Strong rows -- that is the hard decision.\n\n"
    )
    return header + report
