from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

def evaluate_model(model, X_test, y_test, name, output_dir):
    pred = model.predict(X_test)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_classification_report.txt").write_text(
        classification_report(y_test, pred, zero_division=0), encoding="utf-8"
    )
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(13, 11))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return result

def save_comparison(results, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    df.to_csv(out / "metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    df.set_index("model")[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].plot(
        kind="bar", ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("So sánh hiệu năng các mô hình")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=160)
    plt.close(fig)
    return df
