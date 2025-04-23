
```markdown
# 🐶🐱 Dog vs Cat Image Classification using CNN

This project uses a Convolutional Neural Network (CNN) to classify images of dogs and cats. It was developed as part of a learning exercise using the **Kaggle Dogs vs. Cats** dataset and is inspired by the Deeplizard CNN series.

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

| Metric              | Accuracy |
|---------------------|----------|
| **Training Accuracy** | ~98%     |
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
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Jupyter Notebook and run all cells in:
   ```
   Pet_Classification_CNN.ipynb
   ```

---

## 📚 References

- Deeplizard CNN Tutorial: [Watch on YouTube](https://deeplizard.com/learn/video/LhEMXbjGV_4)
- Kaggle Dataset: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

---

## 📌 Future Improvements

- Add data augmentation to improve generalization
- Use transfer learning with pre-trained models like VGG16 or ResNet
- Explore hyperparameter tuning for better performance

---
