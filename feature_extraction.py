import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE




def enhance_contrast(gray_image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    return enhanced_image

def remove_hair(image):
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    image_no_hair = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return image_no_hair

# Preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256))
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray_image = enhance_contrast(gray_image)
    gray_image = remove_hair(gray_image)
    return image_resized, gray_image

# Segment the lesion
def segment_lesion(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_mask

def extract_haralick_features(gray_image):
    glcm = graycomatrix(gray_image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
    features = {
        "contrast": np.mean(graycoprops(glcm, 'contrast')),
        "dissimilarity": np.mean(graycoprops(glcm, 'dissimilarity')),
        "homogeneity": np.mean(graycoprops(glcm, 'homogeneity')),
        "energy": np.mean(graycoprops(glcm, 'energy')),
        "correlation": np.mean(graycoprops(glcm, 'correlation')),
        "ASM": np.mean(graycoprops(glcm, 'ASM'))
    }
    return features

def extract_color_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv_image[:, :, 0])
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])
    return {"mean_hue": mean_hue, "mean_saturation": mean_saturation, "mean_value": mean_value}


# Extract shape features
def extract_shape_features(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
        return {"area": area, "perimeter": perimeter, "circularity": circularity}
    return {"area": 0, "perimeter": 0, "circularity": 0}

# Extract texture features
def extract_texture_features(gray_image):
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return {"contrast": contrast, "energy": energy, "homogeneity": homogeneity}

# Process the dataset
def process_dataset(dataset_path):
    data = []
    for label in ["melanoma", "nevus", "seborrheic_keratosis"]:
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            print(f"Processing {filename}... {label}")
            file_path = os.path.join(label_path, filename)
            try:
                _, gray_image = preprocess_image(file_path)
                binary_mask = segment_lesion(gray_image)
                shape_features = extract_shape_features(binary_mask)
                texture_features = extract_texture_features(gray_image)
                if label == "melanoma":
                    label_w = 1
                else:
                    label_w = 0
                features = {**shape_features, **texture_features, "label": label_w}
                data.append(features)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    return pd.DataFrame(data)

# Main workflow
def main():
    train_path = "skin-lesions/train"
    test_path = "skin-lesions/test"

    # Process train and test datasets
    train_data = process_dataset(train_path)
    test_data = process_dataset(test_path)

    # Balance classes using SMOTE
    smote = SMOTE(random_state=42)
    train_features, train_labels = train_data.drop(columns=['label']), train_data['label']
    train_features, train_labels = smote.fit_resample(train_features, train_labels)
    train_data = pd.concat([train_features, train_labels], axis=1)

    # Save to CSV for further analysis
    train_data.to_csv("train_features.csv", index=False)
    test_data.to_csv("test_features.csv", index=False)

    print("Feature extraction complete! Train and test features saved to CSV.")

# Run the main workflow
main()
