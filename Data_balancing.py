import pandas as pd


input_file = 'input.csv'
df = pd.read_csv(input_file)

# Count the number of rows where isFraud = 1 and where isFraud = 0
fraud_count = df[df['isFraud'] == 1].shape[0]
non_fraud_count = df[df['isFraud'] == 0].shape[0]


rows_to_remove = non_fraud_count - fraud_count

# remove excess
if rows_to_remove > 0:
    non_fraud_indices = df[df['isFraud'] == 0].index[:rows_to_remove]
    df = df.drop(non_fraud_indices)
if rows_to_remove < 0:
    rows_to_remove = rows_to_remove * -1
    fraud_indices = df[df['isFraud'] == 1].index[:rows_to_remove]
    df = df.drop(fraud_indices)

# Write the modified DataFrame to a new CSV file
output_file = 'output.csv'
df.to_csv(output_file, index=False)

print(f"Hakuna matata.")




