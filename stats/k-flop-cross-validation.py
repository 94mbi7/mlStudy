# Import necessary libraries
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist  # Example dataset

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data: Flatten and normalize
X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

# Set up K-Fold Cross-Validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Function to create and compile the model
def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Output layer for classification
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List to store accuracy for each fold
accuracy_per_fold = []

# K-Fold Cross-Validation
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f'Fold {fold + 1}')
    
    # Split the data into training and validation sets for this fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Create a new instance of the model for each fold
    model = create_model()
    
    # Train the model on the training fold
    model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)
    
    # Evaluate the model on the validation fold
    val_predictions = np.argmax(model.predict(X_val_fold), axis=1)
    accuracy = accuracy_score(y_val_fold, val_predictions)
    
    # Store the accuracy for this fold
    accuracy_per_fold.append(accuracy)
    print(f'Accuracy for fold {fold + 1}: {accuracy * 100:.2f}%')

# Calculate the average accuracy across all folds
average_accuracy = np.mean(accuracy_per_fold)
print(f'\nAverage Accuracy Across {k} Folds: {average_accuracy * 100:.2f}%')