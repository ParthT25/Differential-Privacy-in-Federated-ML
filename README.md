# Federated Learning and Differential Privacy for Health Data

This project provides a hands-on, comparative analysis of machine learning models trained on health data using both traditional centralized methods and modern privacy-preserving federated learning techniques. It uses the public Pima Indians Diabetes dataset to demonstrate and evaluate these approaches side-by-side.

The core goal is to explore the trade-offs between model performance, data privacy, and system scalability in different training regimes.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Table of Contents
- [Overview](#overview)
- [Key Concepts Explored](#key-concepts-explored)
- [Models and Experiments](#models-and-experiments)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [How to Run](#how-to-run)
- [Expected Results and Visualizations](#expected-results-and-visualizations)
- [Code Structure](#code-structure)
- [License](#license)

## Overview

This repository contains a Python script (`federated_ml_health.py`) that implements and compares several binary classification models to predict diabetes. The comparisons focus on two primary axes:
1.  **Centralized vs. Federated Training:** How does a model trained on a central server with all data present compare to a model trained in a federated setting where data remains on individual "client" devices?
2.  **Standard vs. Differentially Private Training:** What is the impact on model performance when Differential Privacy (DP) is introduced to provide formal privacy guarantees?

## Key Concepts Explored

- **Centralized Machine Learning:** The classical approach where all data is collected in one place to train a model.
- **Federated Learning (FL):** A decentralized approach where the model is trained across multiple devices (simulated clients) without exchanging the raw data, preserving user privacy. This project uses **TensorFlow Federated (TFF)**.
- **Federated Averaging (FedAvg):** The core algorithm used in this project's FL implementation, where local model updates from clients are aggregated on a central server to produce a new global model.
- **Differential Privacy (DP):** A formal, mathematical guarantee of privacy that makes it possible to learn from aggregate data while limiting what can be learned about any single individual. This is implemented using both **TensorFlow Privacy** (for centralized DP) and TFF's built-in DP aggregators.

## Models and Experiments

The script trains and evaluates the following logistic regression models:

1.  **Centralized `sklearn`:** Trained using various classical optimizers (`liblinear`, `sag`, `lbfgs`).
2.  **Centralized `TensorFlow`:** An equivalent logistic regression model built in Keras, trained on the complete, centralized dataset.
3.  **Federated `TensorFlow Federated` (TFF):** The same Keras model trained using the Federated Averaging algorithm (`FedAvg`).
4.  **Differentially Private Centralized `TensorFlow`:** The centralized TF model trained with a DP-enabled optimizer from the `tensorflow_privacy` library.
5.  **Differentially Private Federated `TFF`:** The federated model trained using a DP-enabled aggregation mechanism (`dp_aggregator`).

## Dataset

The project uses the **Pima Indians Diabetes Database**, a standard dataset for binary classification tasks.
- **Features (8):** Number of pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.
- **Target (1):** Outcome (0 for non-diabetic, 1 for diabetic).

The data is preprocessed using `sklearn.preprocessing.StandardScaler` to normalize the features.

## Prerequisites

You need Python 3.8+ and the libraries listed in `requirements.txt`.

**1. Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Create a `requirements.txt` file** with the following content:
```
nest_asyncio
tensorflow_privacy==0.8.6
tensorflow_federated==0.40.0
pandas
scikit-learn
matplotlib
absl-py
```

**3. Install the dependencies:**
```bash
pip install -r requirements.txt
```

## How to Run

1.  **Save the Code:** Save the provided code as a Python file, for example, `federated_ml_health.py`.

2.  **Activate your virtual environment:**
    ```bash
    source venv/bin/activate
    ```

3.  **Execute the script from your terminal:**
    ```bash
    python federated_ml_health.py
    ```

The script will run all the training and evaluation steps, printing progress and results to the console. At the end, it will display the visualization plots in separate windows.

## Expected Results and Visualizations

The script will produce several key outputs:
- **Console Logs:** Detailed logs for each training round, including metrics like loss and AUC for both centralized and federated models.
- **Plot Windows:** At the end of the execution, several plot windows will appear showing:
    1.  **ROC Curve Comparison:** A plot showing the ROC curves and AUC scores for all models, allowing for direct comparison.
    2.  **AUC vs. Client Participation:** A plot showing how the federated model's AUC score changes as more clients participate in training.
    3.  **Performance and Scalability Benchmarks:** A heatmap and a line plot illustrating how the TFF training runtime scales with the number of clients and features.

### Saving Plots to a File (Optional)

If you prefer to save the plots as image files instead of displaying them in a window, you can modify the last lines of the script. Find each instance of `plt.show()` and replace it with `plt.savefig('descriptive_name.png')`.

For example, change this:
```python
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```
To this:
```python
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_curve_comparison.png')
```

## Code Structure

The Python script is organized into the following logical sections:

- **Imports and Setup:** Installs dependencies and imports all necessary libraries.
- **Data Initialization:** Functions to load the Pima Diabetes dataset and its labels.
- **Centralized `sklearn` Training:** Trains and evaluates baseline `LogisticRegression` models.
- **Centralized `TensorFlow` Training:** Defines and trains a Keras model on the full dataset.
- **Federated `TensorFlow Federated` Training:** Sets up the TFF simulation, builds the `FedAvg` iterative process, and trains the federated model.
- **Federated Training with Differential Privacy:** Modifies the TFF process to use a `dp_aggregator` for private model updates.
- **Centralized Training with Differential Privacy:** Implements a custom DP optimizer from `tensorflow_privacy` and trains the centralized Keras model with it.
- **Results and Visualization:** Code to generate and display the final comparative plots.
- **Performance Benchmarking:** Runs synthetic benchmarks to measure how TFF training time scales with the number of clients and features.

## License

This project is licensed under the Apache License, Version 2.0.
