import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load dataset ---
DATASET_PATH = "dataset_B_FacialImages"
IMG_SIZE = 100

data = []
labels = []

# Open eyes → 0
for img_name in os.listdir(os.path.join(DATASET_PATH, "OpenFace")):
    img_path = os.path.join(DATASET_PATH, "OpenFace", img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)

# Closed eyes → 1
for img_name in os.listdir(os.path.join(DATASET_PATH, "ClosedFace")):
    img_path = os.path.join(DATASET_PATH, "ClosedFace", img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)

data = np.array(data)
labels = np.array(labels)

print("Total samples:", len(data))

# --- Split dataset ---
X_train, X_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# --- HOG feature extraction ---
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

print("HOG feature length:", X_train_hog.shape[1])

# --- PCA ---
pca = PCA(n_components=0.98)
X_train_pca = pca.fit_transform(X_train_hog)
X_val_pca = pca.transform(X_val_hog)
X_test_pca = pca.transform(X_test_hog)

# --- KNN Classifier ---
# Reasonable default: k=5
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train_pca, y_train)

# --- Validation Evaluation ---
y_pred_val = knn.predict(X_val_pca)

print("KNN Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
print("Classification Report:\n", classification_report(y_val, y_pred_val))

# --- Misclassified images visualization ---
misclassified_idx = np.where(y_val != y_pred_val)[0]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified_idx[:12]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_val[idx], cmap='gray')
    plt.title(f"True: {'Open' if y_val[idx]==0 else 'Closed'}\nPred: {'Open' if y_pred_val[idx]==0 else 'Closed'}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Total misclassified (validation): {len(misclassified_idx)}")
