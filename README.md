# Customer_Churn_prediction

## 1. Objective
You aimed to build a binary classification model using PyTorch to predict whether a customer will churn (leave the service) or not, based on features from the Telco Customer Churn dataset.

## 2. Data Preprocessing
### Steps:
Loaded the dataset using pandas.

Dropped customerID (irrelevant to prediction).

Replaced blank spaces with NaN and dropped missing values.

Converted TotalCharges from string to numeric.

Encoded categorical columns using LabelEncoder.

Encoded the target column (Churn) into binary:
'Yes' → 1 and 'No' → 0

Standardized features using StandardScaler.

### Reason:
Neural networks require numerical and scaled inputs.

Categorical encoding enables model interpretability.

Data cleaning ensures integrity for learning.

## 3. Dataset Splitting
Used train_test_split() to split the dataset into:

80% training

20% testing

To assess how well the model generalizes to unseen data.

## 4. Model Architecture
### Structure:
A 4-layer feedforward neural network:

scss
Copy
Edit
Input → Linear(64) → ReLU → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid
### Reason:
ReLU helps capture nonlinear relationships.

Sigmoid ensures outputs are in [0,1] → interpretable as probabilities for binary classification.

## 5. Training
### Settings:
Loss Function: BCELoss (Binary Cross Entropy)

Optimizer: Adam (learning rate = 0.001)

Epochs: 50

Batch size: 32

### Loss Trend:
Epoch 1 → Loss: 0.4881
...
Epoch 25 → Loss: 0.3486
...
Epoch 50 → Loss: 0.2764
### Interpretation:
The training loss consistently decreased, indicating the model learned meaningful patterns.

There was no overfitting observed from loss progression alone.

## 6. Evaluation on Test Data
### Metrics:
Test Accuracy: 75.69%
### Classification Report:
Averages:
Accuracy: 76%

Macro Avg (Unweighted): F1 = 68%

Weighted Avg (By Support): F1 = 76%

### Interpretation:
The model is very good at predicting non-churn customers (Class 0).

However, performance on churned customers (Class 1) is much lower:

Recall = 52%: Nearly half the churn cases are missed.

Precision = 54%: Nearly half of predicted churns are incorrect.

This indicates a class imbalance problem, which the model hasn't fully overcome.

## 7. Model Saving
To reuse or deploy the trained model without retraining.del later without retraining.
