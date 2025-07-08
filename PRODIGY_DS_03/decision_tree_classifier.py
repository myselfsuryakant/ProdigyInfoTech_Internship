import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# IMPORTANT: The separator for this dataset is a semicolon (';')
df = pd.read_csv('bank-additional-full.csv', sep=';')

# --- Initial Data Inspection & Understanding ---
print("--- Initial Data Info ---")
df.info()

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Missing Values (if any) ---")
# Use .any() to quickly see if any column has *any* missing values
print(df.isnull().sum())

print("\n--- Target Variable Distribution ('y') ---")
# This shows how many 'yes' (subscribed) vs 'no' (not subscribed)
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True)) # To see percentages

# --- Data Preprocessing ---

# 1. Drop the 'duration' column
df = df.drop('duration', axis=1)
print("\n--- DataFrame after dropping 'duration' column ---")
print(df.head())
print(f"New DataFrame shape: {df.shape}")

# 2. Identify and inspect categorical columns with 'unknown' values
# Get list of object (categorical) columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
# Exclude the target variable 'y' for now
categorical_features = [col for col in categorical_cols if col != 'y']

print("\n--- Unique values in categorical columns (showing 'unknown' if present) ---")
for col in categorical_features:
    print(f"\nColumn '{col}':")
    print(df[col].value_counts()) # Show counts of each category
    # If there are too many unique values, you might want to only show top N or unique()
    # print(df[col].unique()) # Show all unique values

# 3. Map the target variable 'y' to numerical (0 and 1)
df['y'] = df['y'].map({'yes': 1, 'no': 0})
print("\n--- Target variable 'y' after mapping ---")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True))

# 4. One-Hot Encode the remaining categorical features
# This will create new columns for each category and drop the original categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
# drop_first=True helps avoid multicollinearity by dropping the first category of each feature

print("\n--- DataFrame after One-Hot Encoding and 'y' mapping ---")
print(df_encoded.head())
print(f"New DataFrame shape after encoding: {df_encoded.shape}")
print(df_encoded.info())

# --- Model Building and Evaluation ---

# 1. Separate Features (X) and Target (y)
X = df_encoded.drop('y', axis=1) # All columns except 'y' are features
y = df_encoded['y']             # 'y' is the target variable

print("\n--- Shape of X (Features) and y (Target) ---")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 2. Split Data into Training and Testing Sets
# We'll use 80% for training and 20% for testing.
# stratify=y is important for imbalanced datasets to maintain class distribution in splits.
# random_state ensures reproducibility of your split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Training and Testing Set Shapes ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("\n--- Target distribution in Training set ---")
print(y_train.value_counts(normalize=True))
print("\n--- Target distribution in Testing set ---")
print(y_test.value_counts(normalize=True))


# 3. Build and Train the Decision Tree Classifier
# Initialize the Decision Tree Classifier
# random_state for reproducibility
# You can add parameters like max_depth, min_samples_leaf for tuning later
dt_classifier = DecisionTreeClassifier(random_state=42)

print("\n--- Training Decision Tree Classifier ---")
dt_classifier.fit(X_train, y_train)
print("Decision Tree Classifier trained successfully!")

# 4. Make Predictions
y_pred = dt_classifier.predict(X_test)

# 5. Evaluate the Model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

