import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Sample Data: Hypothetical Age Distribution ---
# In a real-world scenario, you would load your data here.
# For this example, we'll create a sample dataset.
data = {
    'Age Group': ['0-14', '15-24', '25-54', '55-64', '65+'],
    'Population': [18000000, 12000000, 35000000, 15000000, 8000000]
}
df = pd.DataFrame(data)


# --- Create the Bar Chart ---
plt.figure(figsize=(10, 6))
sns.barplot(x='Age Group', y='Population', data=df, palette='viridis')

# --- Add Labels and Title ---
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Population (in millions)', fontsize=12)
plt.title('Distribution of Population by Age Group', fontsize=15)
plt.xticks(rotation=45) # Rotate x-axis labels for better readability
plt.tight_layout() # Adjust layout to make room for rotated labels

# --- Display the Plot ---
plt.show()