# 🐶🐱 Dog vs Cat Image Classification using CNN

![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-blue)
![CNN](https://img.shields.io/badge/Model-CNN-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Dogs%20vs%20Cats-orange)
![Image Classification](https://img.shields.io/badge/Task-Image%20Classification-yellowgreen)
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

This project uses a **Convolutional Neural Network (CNN)** to classify images of dogs and cats. It was developed as part of a learning exercise using the **Kaggle Dogs vs. Cats** dataset and is inspired by the Deeplizard CNN series.

---

## 📁 Dataset

The dataset is sourced from the Kaggle competition:  
👉 [Dogs vs. Cats - Kaggle Competition](https://www.kaggle.com/c/dogs-vs-cats/data)

---

## 🚀 Project Overview

- **Preprocessing**: Images were normalized using `preprocess_input` from `keras.applications.vgg16`, applied via `ImageDataGenerator`.
- **Model Architecture**:
  - 2 × `Conv2D` layers with ReLU activation
  - 1 × `Flatten` layer
  - 1 × `Dense` layer with `Softmax` activation for classification
- **Loss Function**: `categorical_crossentropy` (multi-class)
- **Overfitting Handling**: Detected through training metrics; mitigated using `EarlyStopping` based on validation loss.

---

## 📊 Results

| Metric                  | Accuracy |
|-------------------------|----------|
| **Training Accuracy**   | ~98%     |
| **Validation Accuracy** | ~83%     |

*Early overfitting was addressed by implementing early stopping.*

---

## 🧰 Libraries Used

- `TensorFlow` / `Keras`
- `NumPy`
- `Matplotlib`
- `Jupyter Notebook`
- `VGG16` model (for preprocessing)

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Pet_Classification_CNN.git
   cd Pet_Classification_CNN
