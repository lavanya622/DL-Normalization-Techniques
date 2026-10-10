# Normalization and Feature Scaling Techniques

A practical implementation of **Feature Scaling and Normalization techniques** using Python, Scikit-learn, TensorFlow, and Keras. This repository explores how different scaling and normalization methods help prepare data and improve the training of Machine Learning and Deep Learning models.

## 📌 Project Overview

Feature scaling and normalization are important preprocessing techniques used to bring data into suitable numerical ranges or normalize intermediate neural-network activations.

This project covers traditional feature scaling methods and neural-network normalization techniques through hands-on Jupyter Notebook practicals.

## 🎯 Objectives

- Understand the importance of feature scaling in Machine Learning.
- Implement Standardization, Min-Max Scaling, and Robust Scaling.
- Understand Batch Normalization and Layer Normalization.
- Implement Group Normalization and RMSNorm.
- Compare the purpose and working principles of different techniques.
- Visualize model performance and understand the practical results.

## 📂 Project Structure

```text
05-Normalization-Techniques/
│
├── 01-Feature_Scaling.ipynb
│
├── 02-Batch_Normalization.ipynb
│
├── 03-Layer_Normalization.ipynb
│
├── 04-Group_Normalization.ipynb
│
├── 05-RMSNorm.ipynb
│
└── README.md
```

## 1. Feature Scaling Techniques

Feature scaling transforms numerical features to make their scales more comparable. It is especially useful for algorithms that depend on distances, gradients, or feature magnitudes.

### Techniques Covered

**Standardization — StandardScaler**

Transforms features to have approximately zero mean and unit variance.

Formula:

\[
z = \frac{x-\mu}{\sigma}
\]

**Min-Max Scaling — MinMaxScaler**

Rescales values to a specified range, usually 0 to 1.

\[
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

**Robust Scaling — RobustScaler**

Uses the median and interquartile range (IQR), making it less sensitive to outliers than StandardScaler.

\[
x' = \frac{x-\text{Median}}{IQR}
\]

### Example Code

```python
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

data = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "Salary": [20000, 30000, 40000, 50000, 60000]
})

standard_scaled = StandardScaler().fit_transform(data)
minmax_scaled = MinMaxScaler().fit_transform(data)
robust_scaled = RobustScaler().fit_transform(data)

print("Standard Scaling:\n", standard_scaled)
print("\nMin-Max Scaling:\n", minmax_scaled)
print("\nRobust Scaling:\n", robust_scaled)
```

**Important:** Fit the scaler on the training data only, then use that fitted scaler to transform the test data. This helps prevent data leakage.

## 2. Batch Normalization

Batch Normalization normalizes activations using statistics calculated from a mini-batch during training, followed by learnable scaling and shifting.

**Key points:**
- Commonly used in Deep Learning architectures.
- Often helps stabilize and speed up training.
- Its behavior differs between training and inference.

Example using Keras:

```python
from tensorflow.keras.layers import BatchNormalization

layer = BatchNormalization()
```

## 3. Layer Normalization

Layer Normalization normalizes activations across the features of each individual example.

**Key points:**
- Normalization is performed independently for each example.
- Does not rely on statistics across the batch.
- Commonly used in Transformer and sequence-model architectures.

Example using Keras:

```python
from tensorflow.keras.layers import LayerNormalization

layer = LayerNormalization(axis=-1)
```

## 4. Group Normalization

Group Normalization divides channels into groups and normalizes activations within each group.

**Key points:**
- Does not depend on batch-wide statistics.
- Can be useful when training with small batch sizes.
- Requires choosing a valid number of groups for the channel count.

In the practical notebook, Group Normalization is implemented using a custom TensorFlow/Keras layer.

## 5. RMSNorm (Root Mean Square Normalization)

RMSNorm scales activations using their root mean square (RMS), without subtracting the mean.

Formula:

\[
\operatorname{RMS}(x)=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
\]

\[
\operatorname{RMSNorm}(x)=
\frac{x}{\operatorname{RMS}(x)}\odot g
\]

Here, \(g\) represents learnable scaling parameters and \(\epsilon\) is a small constant used for numerical stability.

**Key points:**
- Does not perform mean-centering.
- Uses RMS-based scaling.
- Used in some modern neural-network and Large Language Model architectures.

The practical notebook implements RMSNorm as a custom TensorFlow/Keras layer.

## 📊 Technique Comparison

| Technique | What it normalizes | Main purpose |
|---|---|---|
| StandardScaler | Input features | Zero mean and unit variance |
| MinMaxScaler | Input features | Scale values to a specified range |
| RobustScaler | Input features | Scale using median and IQR |
| Batch Normalization | Activations using batch statistics | Stabilize neural-network training |
| Layer Normalization | Features within each example | Normalize independently of batch statistics |
| Group Normalization | Activations within channel groups | Normalize without batch-wide statistics |
| RMSNorm | Activations using RMS | Scale activations without subtracting the mean |

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

## ⚙️ Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the required notebook and run its cells in sequence.

## 📈 Practical Results

The notebooks explore the implementation and behavior of each technique. Depending on the practical, results may include:

- Comparison of transformed feature values.
- Training and validation loss plots.
- Model performance metrics.
- Comparison between models with and without selected normalization layers.

Actual results depend on the dataset, architecture, hyperparameters, and training configuration.

## 🎓 Learning Outcomes

After completing these practicals, I gained hands-on experience with:

- Feature preprocessing and scaling.
- Standardization, Min-Max Scaling, and Robust Scaling.
- Batch and Layer Normalization.
- Group Normalization and RMSNorm.
- Implementing normalization layers with TensorFlow/Keras.
- Understanding how preprocessing and normalization differ in Machine Learning and Deep Learning.

## 🚀 Conclusion

This project documents my practical learning journey in Feature Scaling and Deep Learning Normalization techniques. It provides a foundation for understanding data preprocessing, neural-network training, and normalization methods used in modern AI systems.

**Learning Journey:** Machine Learning → Deep Learning → Normalization Techniques

**Learning by implementing. Improving through experimentation.**

---

*This repository is part of my hands-on AI and Machine Learning learning journey.*
