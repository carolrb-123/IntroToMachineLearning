import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from skimage.feature import hog

# =====================
# CONFIGURATION
# =====================
DATASET_PATH = "dataset_B_FacialImages"
IMG_SIZE = 100   # images are already 100x100
RANDOM_STATE = 42

# =====================
# LOAD DATASET
# =====================
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

data = np.array(data)
labels = np.array(labels)

print("Total samples:", len(data))
print("Open eyes samples:", np.sum(labels == 0))
print("Closed eyes samples:", np.sum(labels == 1))

# =====================
# TRAIN / VAL / TEST SPLIT
# =====================
X_train, X_temp, y_train, y_temp = train_test_split(
    data,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print("\nDataset split (original):")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# =====================
# DATA AUGMENTATION (Training set only!)
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

        # 3. Small Rotations (±5, ±10 degrees)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        for angle in [-10, -5, 5, 10]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            aug_images.append(rotated)
            aug_labels.append(label)

    return np.array(aug_images), np.array(aug_labels)


X_train_aug, y_train_aug = augment_data(X_train, y_train)

print(f"Augmented Training samples: {len(X_train_aug)} (original: {len(X_train)})")

# =====================
# HOG FEATURE EXTRACTION
# =====================
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True
}

def extract_hog_features(images):
    features = []

    for img in images:
        # Preprocessing
        img = cv2.equalizeHist(img)
        img = img / 255.0

        hog_features = hog(
            img,
            orientations=HOG_PARAMS["orientations"],
            pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
            cells_per_block=HOG_PARAMS["cells_per_block"],
            block_norm=HOG_PARAMS["block_norm"],
            transform_sqrt=HOG_PARAMS["transform_sqrt"]
        )

        features.append(hog_features)

    return np.array(features)

X_train_hog = extract_hog_features(X_train_aug)
X_val_hog   = extract_hog_features(X_val)
X_test_hog  = extract_hog_features(X_test)

print("\nHOG feature vector length:", X_train_hog.shape[1])

# =====================
# PCA DIMENSIONALITY REDUCTION
# =====================
# Increase variance retention to 0.99 to preserve more eye detail
pca = PCA(n_components=0.99, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_hog)
X_val_pca = pca.transform(X_val_hog)
X_test_pca = pca.transform(X_test_hog)

print("PCA components retained:", X_train_pca.shape[1])

# =====================
# SVM CLASSIFIER WITH GRID SEARCH
# =====================
# Tuning SVM around previously good values (C=10, gamma=0.01)
param_grid = {
    "C": [10, 50, 100],
    "gamma": [0.005, 0.01, 0.05],
    "kernel": ["rbf"]
}

print("\nStarting GridSearchCV (targeting higher accuracy)...")
grid_search = GridSearchCV(
    SVC(class_weight="balanced", probability=True),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_pca, y_train_aug)
svm = grid_search.best_estimator_

print("\nBest Parameters found:", grid_search.best_params_)

# =====================
# SAVE MODEL FOR REAL-TIME DETECTION
# =====================
MODEL_PATH = "drowsy_model.pkl"
print(f"Saving model and PCA to {MODEL_PATH}...")
joblib.dump({
    "svm": svm,
    "pca": pca,
    "hog_params": HOG_PARAMS
}, MODEL_PATH)
print("Model saved successfully.")

# =====================
# VALIDATION EVALUATION
# =====================
y_val_pred = svm.predict(X_val_pca)

print("\n===== VALIDATION RESULTS =====")
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# =====================
# FINAL TEST EVALUATION
# =====================
y_test_pred = svm.predict(X_test_pca)

print("\n===== FINAL TEST RESULTS =====")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# =====================
# VISUALIZE MISCLASSIFICATIONS
# =====================
misclassified = np.where(y_test != y_test_pred)[0]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified[:12]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[idx], cmap="gray")
    plt.title(
        f"True: {'Open' if y_test[idx] == 0 else 'Closed'}\n"
        f"Pred: {'Open' if y_test_pred[idx] == 0 else 'Closed'}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Total misclassified test images:", len(misclassified))
