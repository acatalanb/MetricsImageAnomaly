import time, json, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

class MetricsManager:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start_timer(self): self.start_time = time.time()
    def stop_timer(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

    def format_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s" if h else f"{int(m)}m {int(s):.2f}s"

    def calculate_metrics(self, y_true, y_pred_probs):
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_probs]
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }

    def plot_confusion_matrix(self, cm, labels=['Normal', 'Anomaly']):
        # Optimized size for MS Word (6 inches wide is standard page width)
        fig = plt.figure(figsize=(5, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout()
        return fig

    # === UPDATED: Saves raw ROC data + plot (dataset-specific filenames) ===
    # Modified to include model_name in filenames
    def plot_roc_curve(self, y_true, y_probs, dataset_name="unknown", model_name="unknown", save_dir="cache"):
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import RocCurveDisplay

        os.makedirs(save_dir, exist_ok=True) # Ensure directory exists
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        # Save raw data for later side-by-side plotting
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
        json_path = f"{save_dir}/roc_data_{dataset_name}_{model_name}.json" # Modified line
        with open(json_path, 'w') as f:
            json.dump(roc_data, f)

        # Plot single curve
        plt.figure(figsize=(5, 4))
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot(color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.title(f'ROC Curve - {dataset_name} ({model_name}) (AUC = {roc_auc:.4f})') # Modified title
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{save_dir}/roc_curve_{dataset_name}_{model_name}.png", dpi=300, bbox_inches='tight') # Modified line
        plt.close()

        print(f"✅ ROC saved for {dataset_name} ({model_name}) → AUC = {roc_auc:.4f}") # Modified print
        return roc_auc

    def save_training_stats(self, model_name, duration, best_acc, epochs, dataset_name, cache_dir="cache"):
        os.makedirs(cache_dir, exist_ok=True) # Ensure directory exists
        filepath = os.path.join(cache_dir, "training_stats.json")
        all_stats = {}
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    all_stats = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing {filepath}. Starting with empty stats.")
                all_stats = {}

        current_run_key = f"{model_name}_{dataset_name}"
        all_stats[current_run_key] = {
            "model_name": model_name,
            "duration": duration,
            "best_accuracy": best_acc,
            "epochs": epochs,
            "dataset_name": dataset_name
        }
        with open(filepath, 'w') as f:
            json.dump(all_stats, f, indent=4)
        print(f"📈 Training stats saved for {dataset_name} to {filepath}")