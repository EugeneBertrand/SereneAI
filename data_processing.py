# data_processing.py
import pandas as pd  # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
import nemo.collections.data_processing as curator  # Import NeMo Curator for data processing

# Load the combined data from the CSV file
data = pd.read_csv('combined_bioactivity_data.csv')  # CSV path for combined data - Change if necessary

# Use NeMo Curator to process the data
filtered_data = curator.filter_by_confidence(data, threshold=0.9)  # Filter data by confidence threshold
normalized_data = curator.normalize_columns(filtered_data, columns=['molecular_weight', 'logp'])  # Normalize specified columns
balanced_data = curator.balance_classes(normalized_data, target_column='activity_outcome')  # Balance classes in the dataset

# Save processed data to a CSV file
balanced_data.to_csv('balanced_data.csv', index=False)  # Save balanced data to CSV without index

# Split data into training and validation sets (80% training, 20% validation)
train_data, val_data = train_test_split(balanced_data, test_size=0.2, random_state=42)
# Save training and validation sets to CSV files
train_data.to_csv('train_data.csv', index=False)  # Save training data to CSV without index
val_data.to_csv('val_data.csv', index=False)  # Save validation data to CSV without index
