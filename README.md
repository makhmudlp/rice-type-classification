Rice Type Classification — PyTorch (Beginner Machine Learning Project)

This project predicts rice grain type using handcrafted geometric features from the Kaggle Rice Type Classification dataset. The model is built from scratch using PyTorch, without relying on high-level training frameworks — making it a great beginner-friendly demonstration of core ML concepts.

📌 Project Overview

The goal is to classify rice grains based on shape-related measurements such as Area, MajorAxisLength, and MinorAxisLength.

Using these features, we train a binary classifier using a simple neural network (2 linear layers + Sigmoid output).

📂 Project Structure
```python
rice-type-classification/
│
├── RiceTypeClassificationBeginnerML.ipynb   # Full notebook with EDA + training
│
├── src/
│   ├── dataset.py                            # Custom PyTorch Dataset
│   ├── model.py                              # Neural network model
│   ├── train.py                              # Custom training loop
│   └── predict.py                            # Helper for inference
│
├── requirements.txt                          # Required Python packages
└── README.md                                 # Project documentation
```

🧠 Model Architecture
```scss
The classifier is a simple feed-forward neural network:

Input Layer (]10 features)
↓ Linear
Hidden Layer (10 units)
↓ Linear
Output Layer (1 unit, Sigmoid)
```
The model predicts a probability between 0 and 1, which is rounded to produce a class label.

How to Run the Project
```bash
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/rice-type-classification.git
cd rice-type-classification
```
2. Install Dependencies
``` bash
pip install -r requirements.txt
```
4. Train the Model

Make sure your CSV dataset is available, then run:
```bash
python src/train.py
```

The script:

Loads the dataset

Splits into training + validation

Runs a custom training loop

Prints loss/accuracy per epoch

Saves the model as rice_model.pth

Making Predictions

You can make predictions using:

from src.predict import predict_single

features = [area, major_axis, minor_axis, eccentricity, convex_area,
            equiv_diameter, extent, roundness, perimeter]

result = predict_single("rice_model.pth", features)
print(result)


This returns the predicted probability (0 = class A, 1 = class B).

The training loop tracks: Training loss Validation loss Training accuracy Validation accuracy

Dataset

Kaggle dataset:
🔗 https://www.kaggle.com/datasets/mssmartypants/rice-type-classification

Contains 75,000+ rice grain samples with 10 shape descriptors.

🎯 Why This Project Is Useful

This project demonstrates:

✔ End-to-end ML workflow
✔ Data loading with DataLoader
✔ Custom PyTorch Dataset
✔ Neural network implementation
✔ Manual training loop
✔ GPU support (CUDA if available)
✔ Making predictions with a trained model
✔ Good project structure for GitHub portfolios

It is beginner-friendly but also showcases real practical ML engineering.

Contact 
If you have questions or suggestions, feel free to open an Issue or PR.
