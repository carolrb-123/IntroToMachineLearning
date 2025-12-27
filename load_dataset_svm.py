import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np


""" Load dataset from "dataset_B_FacialImages" directory"""
DATASET_PATH = "dataset_B_FacialImages"
IMG_SIZE = 100   # images are already 100x100

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
print("Open eyes samples:", sum(labels == 0))
print("Closed eyes samples:", sum(labels == 1))
"""Split dataset into training (70%) and temp (30%) sets """


# Flatten images
X = data   # keep images intact
y = labels


# 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
# Split temp into validation and test (15% each)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)
"""Verify the splits"""
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("Train open:", sum(y_train == 0), "closed:", sum(y_train == 1))
print("Val open:", sum(y_val == 0), "closed:", sum(y_val == 1))
print("Test open:", sum(y_test == 0), "closed:", sum(y_test == 1))
img = cv2.equalizeHist(img)
img = img / 255.0

# Feature extraction using HOG
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}
def extract_hog_features(images):
    features = []
    for img in images:
        hog_feat = hog(
            img,
            orientations=HOG_PARAMS["orientations"],
            pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
            cells_per_block=HOG_PARAMS["cells_per_block"],
            block_norm=HOG_PARAMS["block_norm"]
        )
        features.append(hog_feat)
    return np.array(features)

X_train_hog = extract_hog_features(X_train)
X_val_hog   = extract_hog_features(X_val)
X_test_hog  = extract_hog_features(X_test)
print("HOG feature vector length:", X_train_hog.shape[1])
# PCA
pca = PCA(n_components=0.95)  # keep 95% variance
X_train_pca = pca.fit_transform(X_train_hog)
X_val_pca = pca.transform(X_val_hog)
X_test_pca = pca.transform(X_test_hog)

# SVM classifier
"""svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_pca, y_train)"""
svm = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01,
    class_weight='balanced'
)

svm.fit(X_train_pca, y_train)


# Validation evaluation
y_pred_val = svm.predict(X_val_pca)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
print("Classification Report:\n", classification_report(y_val, y_pred_val))

# =====================
# FINAL TEST EVALUATION
# =====================
y_pred_test = svm.predict(X_test_pca)

print("\n===== FINAL TEST RESULTS =====")
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))
misclassified_test = np.where(y_test != y_pred_test)[0]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified_test[:12]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_test[idx], cmap='gray')
    plt.title(
        f"True: {'Open' if y_test[idx]==0 else 'Closed'}\n"
        f"Pred: {'Open' if y_pred_test[idx]==0 else 'Closed'}"
    )
    plt.axis('off')

plt.tight_layout()
plt.show()

print("Total misclassified in test set:", len(misclassified_test))

"""
# Visualize misclassified images
misclassified_idx = np.where(y_val != y_pred_val)[0]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified_idx[:12]):  # show first 12 misclassified
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_val[idx], cmap='gray')
    plt.title(f"True: {'Open' if y_val[idx]==0 else 'Closed'}\nPred: {'Open' if y_pred_val[idx]==0 else 'Closed'}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Total misclassified in validation set: {len(misclassified_idx)}")

"""





