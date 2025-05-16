"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score
import numpy as np

def find_latest_version(log_dir):
    """ #Finds the latest training version folder
"""
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if not versions:
        print("[ERROR] No versioned logs found!")
        return None
    latest_version = sorted(versions, key=lambda x: int(x.split("_")[-1]))[-1]
    return os.path.join(log_dir, latest_version)

def plot_logs(dest_path, trainer, y_true, y_pred):
    """ # Plots training metrics, confusion matrix, precision, F1-score, and accuracy.
"""
    metrics_path = os.path.join(trainer.logger.log_dir, "metrics.csv")

    if not os.path.exists(metrics_path):
        print(f"[ERROR] Metrics file not found: {metrics_path}")
        return

    df = pd.read_csv(metrics_path)

    df.rename(columns=lambda x: x.strip(), inplace=True)

    if 'train_loss_epoch' in df.columns:
        df['train_loss_epoch'] = df['train_loss_epoch'].fillna(method='ffill')
    if 'train_acc_epoch' in df.columns:
        df['train_acc_epoch'] = df['train_acc_epoch'].fillna(method='ffill')

    # NEW (more forgiving):
    df = df.fillna(method='ffill')  # smoother graph

    df_avg = df.groupby("epoch").mean().reset_index()

    # Plot Loss and Accuracy
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    if 'train_loss_epoch' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['train_loss_epoch'], label="Train Loss", marker='o')
    if 'val_loss' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['val_loss'], label="Val Loss", marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if 'train_acc_epoch' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['train_acc_epoch'], label="Train Accuracy", marker='o')
    if 'val_acc' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['val_acc'], label="Val Accuracy", marker='o')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    os.makedirs(dest_path, exist_ok=True)
    metrics_plot_path = os.path.join(dest_path, "loss_accuracy_plot.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    # Precision, F1-Score, Accuracy
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    metrics = [precision, f1, accuracy]
    labels = ['Precision', 'F1 Score', 'Accuracy']
    sns.barplot(x=labels, y=metrics)
    plt.title("Precision, F1 Score, and Accuracy")
    plt.ylim(0, 1)
    for idx, value in enumerate(metrics):
        plt.text(idx, value + 0.01, f'{value:.2f}', ha='center')
    metrics_summary_path = os.path.join(dest_path, "precision_f1_accuracy.png")
    plt.savefig(metrics_summary_path)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    confusion_matrix_path = os.path.join(dest_path, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()

    print(f"[✅] Plots saved at: {dest_path}")
    """
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def find_latest_version(log_dir):
    versions = [d for d in os.listdir(log_dir) if d.startswith("version_")]
    if not versions:
        print("[ERROR] No versioned logs found!")
        return None
    latest_version = sorted(versions, key=lambda x: int(x.split("_")[-1]))[-1]
    return os.path.join(log_dir, latest_version)

def plot_logs(trainer, y_true, y_pred, y_probs=None):
    # --- Auto-use versioned log directory ---
    base_dir = trainer.logger.log_dir
    dest_path = os.path.join(base_dir, "plots")
    os.makedirs(dest_path, exist_ok=True)

    metrics_path = os.path.join(base_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"[ERROR] Metrics file not found: {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df = df.fillna(method='ffill')

    df_avg = df.groupby("epoch").mean().reset_index()

    # --- Plot Loss ---
    plt.figure(figsize=(10, 6))
    if 'train_loss' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['train_loss'], label="Train Loss", marker='o')
    if 'val_loss' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['val_loss'], label="Validation Loss", marker='o')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, "train_val_loss.png"))
    plt.close()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 6))
    if 'train_acc' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['train_acc'], label="Train Accuracy", marker='o')
    if 'val_acc' in df_avg.columns:
        plt.plot(df_avg['epoch'], df_avg['val_acc'], label="Validation Accuracy", marker='o')
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, "train_val_accuracy.png"))
    plt.close()

    # --- Precision, F1, Accuracy Bar Chart ---
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    metrics = [precision, f1, accuracy]
    labels = ['Precision', 'F1 Score', 'Accuracy']
    sns.barplot(x=labels, y=metrics)
    plt.title("Precision, F1 Score, and Accuracy")
    plt.ylim(0, 1)
    for idx, value in enumerate(metrics):
        plt.text(idx, value + 0.01, f'{value:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, "precision_f1_accuracy.png"))
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, "confusion_matrix.png"))
    plt.close()

    # --- ROC Curve ---
    if y_probs is not None:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.5)")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(dest_path, "roc_curve.png"))
        plt.close()
        print("[✅] ROC curve saved.")
    else:
        print("[⚠️] y_probs not provided — skipping ROC curve.")

    print(f"[✅] All plots saved to: {dest_path}")
