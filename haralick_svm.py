import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import graycomatrix, graycoprops

# Haralick Feature Extraction
def extract_haralick_features(gray_image):
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, asm]

# Prepare data for classification
def prepare_data_with_haralick(feature_file):
    data = pd.read_csv(feature_file)

    # Extract Haralick features from raw images (assuming `gray_image_path` exists in data)
    # Here, gray_image_path is the column containing paths to grayscale images
    # Uncomment below lines if implementing end-to-end with image paths
    # data['haralick_features'] = data['gray_image_path'].apply(lambda x: extract_haralick_features(cv2.imread(x, cv2.IMREAD_GRAYSCALE)))

    # Flatten Haralick features into separate columns (assuming 6 features)
    # Uncomment and replace `[0, 0, 0, 0, 0, 0]` with extracted haralick features for integration
    haralick_features = pd.DataFrame(data.apply(lambda row: [0, 0, 0, 0, 0, 0], axis=1).tolist(), 
                                     columns=['har_contrast', 'har_dissimilarity', 'har_homogeneity', 
                                              'har_energy', 'har_correlation', 'har_asm'])
    
    # Combine Haralick features with existing features
    data = pd.concat([data, haralick_features], axis=1)

    # Encode labels (benign = 0, malignant = 1)
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])

    # Separate features and labels
    X = data.drop(columns=['label'])
    y = data['label']

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# SVM with k-Fold Cross-Validation
def svm_with_kfold(X, y, n_splits=5):
    # Define SVM model
    svm_model = SVC(kernel='linear', C=1, random_state=42)

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

# Main Workflow
def main():
    # Paths to the feature files
    train_file = "train_features.csv"

    # Prepare data with Haralick features
    X, y = prepare_data_with_haralick(train_file)

    # Train and evaluate SVM with k-fold cross-validation
    svm_with_kfold(X, y, n_splits=5)

# Run the main workflow
main()
