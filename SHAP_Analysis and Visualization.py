import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the CSV file
shap_values = pd.read_csv('shap_values.csv')

# Calculating the absolute SHAP values for each feature
abs_shap_values = shap_values.abs()

# Displaying the first rows of the DataFrame to understand its structure
print(shap_values.head())

# Creating a boxplot to visualize the distribution of SHAP values
plt.figure(figsize=(12, 6))
sns.boxplot(data=abs_shap_values, orient='h')
plt.title('Distribution of SHAP Values for Features')
plt.xlabel('SHAP Values')
plt.ylabel('Features')
plt.grid(True)
plt.savefig('shap_values_distribution.pdf')  # Saving the figure
plt.show()

# Calculating the mean SHAP values and sorting them
mean_shap = abs_shap_values.mean().sort_values(ascending=False)

# Creating a bar plot with the mean SHAP values
plt.figure(figsize=(12, 6))
mean_shap.plot(kind='bar')
plt.title('Average Contribution of Features to Model Predictions')
plt.xlabel('Features')
plt.ylabel('Mean SHAP Values')
plt.grid(axis='y')
plt.savefig('mean_feature_contribution.pdf')  # Saving the figure
plt.show()
