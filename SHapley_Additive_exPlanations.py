import os
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time  # Importing the module for time tracking

# Start measuring time
start_time = time.time()

# Path to the saved model and scaler
model_path = os.path.join("Models", "Optimised Gradient boosting_v2.pkl")
scaler_path = os.path.join("Models", "Optimised Gradient boosting_scaler_v2.pkl")

# Path to the data file
data_file_path = os.path.join("CSV_files", "data_P3.csv")

# Path for saving SHAP results
shap_output_directory = "."

# Load the saved model and scaler
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the dataset
data = pd.read_csv(data_file_path)

# Input features
feature_names = ["Fund_Freq1", "Fund_Freq2", "PGA_of_the_recording_scale_1", 
                 "PGA_of_the_recording_scale_2", "b_Hieght", "dim_x", "dim_y",
                 "b_st", "h_st", "b_gr", "h_gr", "E", "MstY", "MstX", 
                 "Mgr", "Lshape", "bay2", "no_span_2", "no_bay_2", 
                 "no_story_2", "T1", "T2", "T3"]

# Prepare the data: apply the scaler and select only the feature columns
X = data[feature_names]
y = data["DI_cladire"]  # assuming this is the target column
X_scaled = scaler.transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the SHAP explainer for the Gradient Boosting model
explainer = shap.Explainer(model)

# Compute SHAP values on the testing set
shap_values = explainer(X_test)

# Visualize the global feature importance (summary_plot)
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

# Save the figure with the global feature importance
summary_plot_path = os.path.join(shap_output_directory, "shap_summary_plot.png")
plt.savefig(summary_plot_path, bbox_inches='tight')
print(f"Summary plot saved at: {summary_plot_path}")

# Check if there are enough rows in X_test to create a force_plot
if X_test.shape[0] > 1:
    # Visualize the impact of each feature on a specific prediction (force_plot)
    # Use shap_values[0] because we are in the regression case
    shap.force_plot(explainer.expected_value, shap_values.values[0], X_test[0], feature_names=feature_names, show=False)

    # Save the force plot for a random row
    force_plot_path = os.path.join(shap_output_directory, "shap_force_plot.html")
    shap.save_html(force_plot_path, shap.force_plot(explainer.expected_value, shap_values.values[0], X_test[0], feature_names=feature_names))
    print(f"Force plot saved at: {force_plot_path}")
else:
    print(f"The testing set has only {X_test.shape[0]} row(s). Cannot generate force plot.")

# Save the SHAP values to a CSV for further analysis
shap_values_csv_path = os.path.join(shap_output_directory, "shap_values.csv")
shap_values_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_values_df.to_csv(shap_values_csv_path, index=False)
print(f"SHAP values have been saved to: {shap_values_csv_path}")

# Calculate the total execution time
end_time = time.time()
execution_time = end_time - start_time

# Convert time from seconds to minutes and seconds
minutes = execution_time // 60
seconds = execution_time % 60

print(f"The simulation took: {minutes:.0f} minutes and {seconds:.2f} seconds")
