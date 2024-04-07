import pandas as pd

# Load the dataset
file_path = 'train.csv'
df = pd.read_csv(file_path)

# Drop the original date/time column
df = df.drop('dateOfBirth', axis=1)
df = df.drop('transNum', axis=1)

# Display the initial number of rows for comparison
print(f"Initial number of rows: {df.shape[0]}")

# Remove any rows with missing data
df_cleaned = df.dropna()

# Display the number of rows after cleaning
print(f"Number of rows after removing rows with missing data: {df_cleaned.shape[0]}")

# Save the cleaned dataframe back to a new CSV file if needed
cleaned_file_path = 'train_cleaned.csv'  # Update or specify your desired output file path
df_cleaned.to_csv(cleaned_file_path, index=False)
