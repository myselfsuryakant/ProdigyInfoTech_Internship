# Task 03: Decision Tree Classifier

**Description**: Built a Decision Tree Classifier to predict customer subscriptions using the Bank Marketing dataset from UCI. The task involved data preprocessing, model training, and evaluation.

**Files**:
- `decision_tree_classifier.py`: Python script for data loading, preprocessing, training, and evaluation.
- `bank-additional-full.csv`: Dataset used for the task.

**Key Steps Performed**:
1. **Data Loading & Inspection**: Loaded `bank-additional-full.csv` and checked for missing values and data types.
2. **Data Preprocessing**: Dropped the `duration` column, handled `unknown` values, applied one-hot encoding to categorical features, and mapped the target variable `y` (`yes`/`no` to 1/0).
3. **Model Training**: Trained a Decision Tree Classifier with `random_state=42`.
4. **Evaluation**: Evaluated the model using accuracy, classification report, and confusion matrix. Noted class imbalance issues (low recall for `yes` class).
5. **Next Steps**: Suggested improvements like SMOTE for class imbalance and hyperparameter tuning.

**Output**:
- Accuracy: ~0.84 (from your output, Page 130).
- Classification Report: Precision ~0.91, Recall ~0.90, F1-score ~0.91 for class 0; lower for class 1 due to imbalance.

**Future Improvements (Potential Work):**
- Implement techniques to address class imbalance (e.g., SMOTE, `class_weight='balanced'`).
- Perform hyperparameter tuning for the Decision Tree (e.g., `max_depth`, `min_samples_leaf`).
- Experiment with other machine learning models (e.g., Random Forest, XGBoost) which are often more robust for imbalanced datasets.

---

Feel free to explore the code and the results!