from IPython.display import Image, display
import matplotlib.pyplot as plt
import os
import json
import math
import numpy as np

def get_epochs_from_stats(model_name, dataset_name, cache_dir="./cache"):
    filepath = os.path.join(cache_dir, "training_stats.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                all_stats = json.load(f)
            key = f"{model_name}_{dataset_name}"
            if key in all_stats and "epochs" in all_stats[key]:
                return all_stats[key]["epochs"]
        except:
            pass
    return None

def plot_multi_model_confusion_matrices(models, dataset_name="yonsei_faces", cache_dir="./cache"):
    num_models = len(models)
    ncols = 2
    nrows = math.ceil(num_models / ncols)
    # Reduced figsize for MS Word compatibility (8-10 inches total width)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if nrows > 1 or ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, model_name in enumerate(models):
        ax = axes[i]
        img_path = os.path.join(cache_dir, f"confusion_matrix_{dataset_name}_{model_name}.png")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{model_name}', fontsize=12)
        else:
            ax.axis('off')
            ax.set_title(f'{model_name} (Not Found)', fontsize=12)

    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])

    epochs = get_epochs_from_stats(models[0], dataset_name)
    epochs_str = f" ({epochs} epochs)" if epochs else ""
    plt.suptitle(f'Confusion Matrix Comparison on {dataset_name}{epochs_str}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    save_path = os.path.join(cache_dir, f"confusion_matrix_comparison_{dataset_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Multi-model Confusion Matrix saved as {save_path}")

def plot_separate_roc_curves(models, dataset_name="yonsei_faces", cache_dir="./cache"):
    num_models = len(models)
    ncols = 2
    nrows = 2
    # Reduced figsize for MS Word compatibility
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    
    axes = axes.flatten()

    found_any = False
    for i in range(len(axes)):
        ax = axes[i]
        if i < num_models:
            model_name = models[i]
            filepath = os.path.join(cache_dir, f"roc_data_{dataset_name}_{model_name}.json")
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    ax.plot(data['fpr'], data['tpr'], lw=2, color='darkorange')
                    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'{model_name}\nAUC = {data["auc"]:.4f}')
                    ax.grid(True, alpha=0.3)
                    found_any = True
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    ax.axis('off')
                    ax.set_title(f'{model_name} (Error)')
            else:
                print(f"⚠️ ROC data not found for {model_name} ({dataset_name})")
                ax.axis('off')
                ax.set_title(f'{model_name} (Not Found)')
        else:
            fig.delaxes(ax)

    if found_any:
        epochs = get_epochs_from_stats(models[0], dataset_name, cache_dir)
        epochs_str = f" ({epochs} epochs)" if epochs else ""
        plt.suptitle(f'ROC Curves for {dataset_name}{epochs_str}', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        save_path = os.path.join(cache_dir, f"roc_curves_separate_{dataset_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Separate ROC curves saved as {save_path}")
    else:
        plt.close()
        print("❌ No ROC data found for any model. Run evaluation first.")

if __name__ == "__main__":
    models_to_compare = ["DenseNet121", "ResNet50", "EfficientNetB0"]
    # Update default dataset to match current usage in run_all.bat
    current_dataset = "yonsei_faces" 
    
    plot_multi_model_confusion_matrices(models_to_compare, dataset_name=current_dataset)
    plot_separate_roc_curves(models_to_compare, dataset_name=current_dataset)