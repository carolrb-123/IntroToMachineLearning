import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog

# =====================
# CONFIGURATION
# =====================
DATASET_PATH = "dataset_B_FacialImages"
IMG_SIZE = 100
RANDOM_STATE = 42
MODEL_PATH = "drowsy_model.pkl"

# =====================
# LOAD DATASET
# =====================
def load_data():
    data = []
    labels = []

    # Open eyes → label 0
    open_path = os.path.join(DATASET_PATH, "OpenFace")
    for img_name in os.listdir(open_path):
        img_path = os.path.join(open_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(0)

    # Closed eyes → label 1
    closed_path = os.path.join(DATASET_PATH, "ClosedFace")
    for img_name in os.listdir(closed_path):
        img_path = os.path.join(closed_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(1)

    return np.array(data), np.array(labels)

# =====================
# DATA AUGMENTATION
# =====================
def augment_data(images, labels):
    print("Augmenting training data...")
    aug_images = []
    aug_labels = []

    for img, label in zip(images, labels):
        # 1. Original
        aug_images.append(img)
        aug_labels.append(label)

        # 2. Horizontal Flip
        flip_img = cv2.flip(img, 1)
        aug_images.append(flip_img)
        aug_labels.append(label)

        # 3. Small Rotations
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        for angle in [-10, -5, 5, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            aug_images.append(rotated)
            aug_labels.append(label)

    return np.array(aug_images), np.array(aug_labels)

# =====================
# FEATURE EXTRACTION
# =====================
def extract_hog_features(images):
    features = []
    for img in images:
        img = cv2.equalizeHist(img)
        img = img / 255.0
        hog_features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True
        )
        features.append(hog_features)
    return np.array(features)

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    X_train_aug, y_train_aug = X_train, y_train
    print(f"Using {len(X_train_aug)} training samples (no augmentation).")

    print("Extracting HOG features...")
    X_train_hog = extract_hog_features(X_train_aug)
    X_test_hog = extract_hog_features(X_test)

    print("Fitting PCA...")
    pca = PCA(n_components=0.99, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_hog)
    X_test_pca = pca.transform(X_test_hog)
    print(f"PCA components: {X_train_pca.shape[1]}")

    print("Training SVM...")
    svm = SVC(C=10, gamma=0.01, kernel="rbf", class_weight="balanced", probability=True)
    svm.fit(X_train_pca, y_train_aug)

    test_accuracy = svm.score(X_test_pca, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump({
        "svm": svm,
        "pca": pca,
        "hog_params": {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2)
        }
    }, MODEL_PATH)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
