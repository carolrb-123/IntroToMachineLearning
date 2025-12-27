import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================
# Dataset loading
# =====================
DATASET_PATH = "dataset_B_FacialImages"
IMG_SIZE = 100

data = []
labels = []

# Open eyes → 0
for img_name in os.listdir(os.path.join(DATASET_PATH, "OpenFace")):
    img = cv2.imread(os.path.join(DATASET_PATH, "OpenFace", img_name), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.equalizeHist(img)
        img = img / 255.0
        data.append(img)
        labels.append(0)

# Closed eyes → 1
for img_name in os.listdir(os.path.join(DATASET_PATH, "ClosedFace")):
    img = cv2.imread(os.path.join(DATASET_PATH, "ClosedFace", img_name), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.equalizeHist(img)
        img = img / 255.0
        data.append(img)
        labels.append(1)

data = np.array(data)
labels = np.array(labels)

print("Total samples:", len(data))

# =====================
# Train / Val / Test split
# =====================
X_train, X_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, random_state=42, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =====================
# HOG feature extraction
# =====================
def extract_hog(images):
    features = []
    for img in images:
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )
        features.append(feat)
    return np.array(features)

X_train_hog = extract_hog(X_train)
X_val_hog   = extract_hog(X_val)
X_test_hog  = extract_hog(X_test)

print("HOG feature length:", X_train_hog.shape[1])

# =====================
# PCA
# =====================
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_hog)
X_val_pca   = pca.transform(X_val_hog)
X_test_pca  = pca.transform(X_test_hog)

# =====================
# Random Forest classifier
# =====================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_pca, y_train)

# =====================
# Validation evaluation
# =====================
y_pred_val = rf.predict(X_val_pca)

print("RF Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
print("Classification Report:\n", classification_report(y_val, y_pred_val))
misclassified_idx = np.where(y_val != y_pred_val)[0]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(misclassified_idx[:12]):  # show first 12 misclassified
    plt.subplot(3, 4, i + 1)
    plt.imshow(X_val[idx], cmap='gray')
    plt.title(f"True: {'Open' if y_val[idx]==0 else 'Closed'}\nPred: {'Open' if y_pred_val[idx]==0 else 'Closed'}")
    plt.axis('off')

plt.tight_layout()
plt.show()
misclassified = np.where(y_val != y_pred_val)[0]
print("Total misclassified (validation):", len(misclassified))
