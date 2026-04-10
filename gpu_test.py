import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from PIL import Image

# 1. Hardware Check
cuda_ready = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_ready else "None"
print(f"--- Hardware Check ---")
print(f"CUDA Available: {cuda_ready}")
print(f"Using GPU: {gpu_name}\n")

# 2. Data Generation with Numpy & Scikit-Learn
print("Generating synthetic dataset...")
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

# 3. Visualization with Seaborn/Matplotlib
print("Creating visualization...")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title(f"Feature Correlation - Running on {gpu_name}")

# Save the plot (since plt.show() won't work over SSH)
plot_filename = "correlation_map.png"
plt.savefig(plot_filename)
print(f"Visualization saved as: {plot_filename}")

# 4. Image Processing with Pillow
img = Image.open(plot_filename)
print(f"Pillow verified: Image size is {img.size}")

print("\n--- All libraries working perfectly! ---")
