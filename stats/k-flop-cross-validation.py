from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist  

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=42)

def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

accuracy_per_fold = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f'Fold {fold + 1}')
    
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    model = create_model()
    
    model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)
    
    val_predictions = np.argmax(model.predict(X_val_fold), axis=1)
    accuracy = accuracy_score(y_val_fold, val_predictions)
    
    accuracy_per_fold.append(accuracy)
    print(f'Accuracy for fold {fold + 1}: {accuracy * 100:.2f}%')

average_accuracy = np.mean(accuracy_per_fold)
print(f'\nAverage Accuracy Across {k} Folds: {average_accuracy * 100:.2f}%')
