import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- Load Dataset ----------------
data = pd.read_csv(r"C:\Users\harsh\python-projects\Support Vector Machines\data\breast-cancer.csv")
print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# Drop ID column if present
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Encode target (diagnosis: M/B → 1/0)
le = LabelEncoder()
y = le.fit_transform(data["diagnosis"])
X = data.drop("diagnosis", axis=1)

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Linear SVM ----------------
svm_linear = SVC(kernel="linear", C=1, random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)

print("\n📌 Linear SVM Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Classification Report:\n", classification_report(y_test, y_pred_linear))

# ---------------- RBF SVM ----------------
svm_rbf = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

print("\n📌 RBF SVM Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Classification Report:\n", classification_report(y_test, y_pred_rbf))

# ---------------- Hyperparameter Tuning ----------------
param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.001], "kernel": ["rbf"]}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print("\n📌 Best Parameters (RBF SVM):", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# ---------------- Cross-validation ----------------
cv_scores = cross_val_score(svm_rbf, X_train_scaled, y_train, cv=5)
print("\n📌 Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix - RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ---------------- Decision Boundary Plot ----------------
def plot_svm_boundary(model, X, y, title, filename):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Use only first 2 features for visualization
svm_2d = SVC(kernel="linear", C=1).fit(X_train_scaled[:, :2], y_train)
plot_svm_boundary(svm_2d, X_train_scaled[:, :2], y_train,
                  "Linear SVM Decision Boundary (2 features)",
                  "outputs/linear_svm_boundary.png")

svm_2d_rbf = SVC(kernel="rbf", C=1, gamma="scale").fit(X_train_scaled[:, :2], y_train)
plot_svm_boundary(svm_2d_rbf, X_train_scaled[:, :2], y_train,
                  "RBF SVM Decision Boundary (2 features)",
                  "outputs/rbf_svm_boundary.png")

print("\n✅ All results and plots saved in outputs/")
