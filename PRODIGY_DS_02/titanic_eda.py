import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Data ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv') # Load test_df too, though focus EDA on train_df

print("--- Initial Inspection ---")
print("Train DataFrame Info:")
train_df.info()
print("\nTrain DataFrame Head:")
print(train_df.head())
print("\nMissing values in Train DataFrame:")
print(train_df.isnull().sum())
print("\nDescriptive Statistics for Train DataFrame:")
print(train_df.describe())

# You can do similar for test_df but mostly focus on train_df for survival analysis

# --- 2. Data Cleaning ---

# Handle Missing 'Age' values (e.g., fill with median)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True) # Don't forget test_df if you'll use it later

# Handle Missing 'Embarked' values (e.g., fill with mode)
# Find mode of 'Embarked' in train_df
most_frequent_embarked = train_df['Embarked'].mode()[0]
train_df['Embarked'].fillna(most_frequent_embarked, inplace=True)

# Handle Missing 'Fare' values in test_df (it usually has one missing value)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Drop 'Cabin' column due to many missing values (or extract info if you prefer)
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

# (Optional) Check for duplicates, but usually not an issue in Titanic
# print("\nDuplicates in Train DataFrame:", train_df.duplicated().sum())

print("\n--- After Cleaning Missing Values (Train DataFrame) ---")
print(train_df.isnull().sum())

# --- 3. Exploratory Data Analysis (EDA) ---

# Univariate Analysis: Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png') # Save the plot!
plt.show() # Display the plot

# Univariate Analysis: Survival Count
plt.figure(figsize=(7, 5))
sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.savefig('survival_count.png')
plt.show()

# Bivariate Analysis: Survival by Sex
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.savefig('sex_survival_barplot.png') # This is one you linked in README!
plt.show()

# Bivariate Analysis: Survival by Pclass
plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.savefig('pclass_survival_barplot.png') # This is one you linked in README!
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation calculation
numeric_cols = train_df.select_dtypes(include=['number'])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png') # This is one you linked in README!
plt.show()

# Continue with other plots as per your EDA plan...
# e.g., Age distribution vs. Survived (violin plot or box plot)
# e.g., Fare distribution vs. Survived
# e.g., Embarked vs. Survived
# e.g., Create FamilySize and IsAlone features and plot their relationship with Survival