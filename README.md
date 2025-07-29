# Customer_Churn_prediction

# Step-by-Step Project Explanation
# 1. Problem Definition
Customer churn prediction is a binary classification task. The goal is to predict whether a customer will churn (leave) or stay based on their service usage, demographics, and account details.

# 2. Data Loading and Exploration
The dataset is sourced from a telecom company and includes features like gender, contract type, internet service, payment method, tenure, etc.

The target variable is "Churn" with values Yes or No.

# 3. Data Cleaning
Some columns may contain missing or inconsistent values (like spaces in numeric fields).

These are replaced with NaN and dropped to ensure the model isn't misled by invalid inputs.

# 4. Target Encoding
The target variable "Churn" is converted from Yes/No to 1/0 to make it suitable for machine learning models.

# 5. Categorical Feature Encoding
Categorical (text) columns are converted to numerical values using Label Encoding.

This allows neural networks to process inputs that were originally in string format.

# 6. Feature Scaling
Input features are standardized using StandardScaler, which transforms data to have a mean of 0 and standard deviation of 1.

This ensures the model trains efficiently and avoids biases due to varying scales of input features.

# 7. Train-Test Split
The dataset is split into a training set and a test set.

The model is trained on the training data and evaluated on the unseen test data to validate its generalization.

# 8. Tensor Conversion
The NumPy arrays (from the processed DataFrame) are converted into PyTorch tensors.

The target tensors are reshaped into 2D format to match the expected output shape of the model.

# 9. Dataset and DataLoader Creation
PyTorch's TensorDataset wraps the input and target tensors.

The DataLoader helps in batching the data and shuffling it during training for better generalization and performance.

# 10. Model Definition
A simple feedforward neural network is defined using nn.Module.

It includes multiple Linear (fully connected) layers with ReLU activation functions, and ends with a Sigmoid layer to output a probability between 0 and 1.

# 11. Loss Function and Optimizer
The model uses Binary Cross-Entropy Loss (BCELoss) because the task is binary classification.

The optimizer used is Adam, which adapts learning rates during training for faster convergence.

# 12. Training the Model
The model is trained for a fixed number of epochs.

In each epoch:

A forward pass computes predictions.

The loss is calculated.

A backward pass computes gradients.

The optimizer updates model weights using the gradients.

# 13. Model Evaluation
The trained model is evaluated on the test set.

The predicted probabilities are converted into binary predictions using a threshold (e.g., 0.5).

Metrics like accuracy and a classification report (precision, recall, F1-score) are used to measure performance.

14. Saving the Model
The trained model's parameters (weights and biases) are saved to a .pth file.

This allows reusing the model later without retraining.
