import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.1, 1]
}


# Load the features
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

# Prepare data for classification
def prepare_data(train_data, test_data):
    # Encode labels (benign = 0 and seborrheic_keratosis = 0, melanoma = 1)
    le = LabelEncoder()
    train_data['label'] = le.fit_transform(train_data['label'])
    test_data['label'] = le.transform(test_data['label'])

    # Separate features and labels
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']
    X_test = test_data.drop(columns=['label'])
    y_test = test_data['label']

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test


# Train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Train a logistic regression model
    model = LogisticRegression  (C=1, random_state=42, max_iter=1000, solver='liblinear', penalty='l2', tol=0.0001)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def svm_with_kfold(X, y, n_splits=5):
    # Define SVM model
    svm_model = SVC(kernel='rbf', C=10, random_state=42, gamma=1)

    # Perform Stratified k-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(svm_model, X, y, cv=skf, scoring='accuracy')

    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")

    # Train on the full dataset and evaluate
    svm_model.fit(X, y)
    y_pred = svm_model.predict(X)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Main workflow
def main():
    # Paths to the feature files
    train_file = "train_features.csv"
    test_file = "test_features.csv"

    # Load data
    train_data, test_data = load_data(train_file, test_file)

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data)

    # Train and evaluate
    #train_and_evaluate(X_train, y_train, X_test, y_test)

    

    # Merge train and test data for cross-validation
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    
    # Train and evaluate SVM with k-fold cross-validation
    svm_with_kfold(X, y, n_splits=5)

    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best Cross-Validation Score: {grid.best_score_:.2f}")

# Run the main workflow
main()
