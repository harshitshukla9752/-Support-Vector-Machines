# SVM Classification Project

## Task 7: Support Vector Machines (SVM)

### Objective

Use SVMs for linear and non-linear classification.

### Tools

* Python
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn

### Dataset

* Dataset: Breast Cancer Dataset
* Path: `data/breast_cancer.csv`
* Target column: `diagnosis` (converted to 0/1)
* Features: All remaining columns except `id` and `diagnosis`

### Steps Covered

1. **Load and prepare a dataset for binary classification**

   * Load CSV using Pandas
   * Encode target labels using LabelEncoder
   * Split dataset into train/test sets

2. **Train an SVM with linear and RBF kernel**

   * Linear SVM using `kernel='linear'`
   * RBF SVM using `kernel='rbf'`

3. **Visualize decision boundary using 2D data**

   * Function `plot_svm_boundary` plots decision boundaries for 2 features
   * Saved plots in `outputs/linear_svm_boundary.png` and `outputs/rbf_svm_boundary.png`

4. **Tune hyperparameters like C and gamma**

   * Grid search with `C` and `gamma` values
   * Best parameters printed

5. **Use cross-validation to evaluate performance**

   * `cross_val_score` used
   * Mean CV accuracy printed

6. **Evaluation Metrics**

   * Accuracy
   * Classification report
   * Confusion matrix

### Output

* **Linear SVM**: Accuracy, classification report, decision boundary plot
* **RBF SVM**: Accuracy, classification report, decision boundary plot, best hyperparameters
* **Cross-validation**: Mean CV accuracy
* **Confusion Matrix**: Heatmap saved in outputs folder

### What You'll Learn

* Margin maximization
* Kernel trick
* Hyperparameter tuning
* Model evaluation with cross-validation

### How to Run

1. Ensure all dependencies are installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Run the script:

```bash
python src/svm_classifier.py
```

3. Outputs and plots will be saved in the `outputs/` folder.
