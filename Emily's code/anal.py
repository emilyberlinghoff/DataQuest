from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

# Custom function to handle unseen labels during transformation
def safe_transform(column, trained_encoder):
    unseen_label = -1  # Assign -1 for unseen labels
    return column.map(lambda x: trained_encoder.transform([x])[0] if x in trained_encoder.classes_ else unseen_label)

# Load the dataset
df = pd.read_csv('train_cleaned.csv')

# Convert the date/time column to datetime and extract components
df['transDate'] = pd.to_datetime(df['transDate'])
df['year'] = df['transDate'].dt.year
df['month'] = df['transDate'].dt.month
df['day'] = df['transDate'].dt.day
df.drop('transDate', axis=1, inplace=True)

# Initialize LabelEncoders
label_encoders = {}
categorical_cols = ['category', 'city', 'job', 'state', 'business', 'firstName', 'lastName', 'street']

# Encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encoding for 'gender'
df = pd.get_dummies(df, columns=['gender'])

# Separate features and target variable
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoders
dump(model, 'random_forest_model.joblib')
for col, le in label_encoders.items():
    dump(le, f'{col}_encoder.joblib')

# Predict on the test set and display performance
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Load and preprocess new data
test_df = pd.read_csv('test.csv')
test_df['transDate'] = pd.to_datetime(test_df['transDate'])
test_df['year'] = test_df['transDate'].dt.year
test_df['month'] = test_df['transDate'].dt.month
test_df['day'] = test_df['transDate'].dt.day
test_df.drop('transDate', axis=1, inplace=True)

# Apply saved encoders to the test set
for col in categorical_cols:
    le = load(f'{col}_encoder.joblib')
    test_df[col] = safe_transform(test_df[col], le)

# One-hot encoding for 'gender', ensuring alignment with the training data
test_df = pd.get_dummies(test_df, columns=['gender'])
required_columns = X_train.columns.tolist()
missing_cols = set(required_columns) - set(test_df.columns)
for column in missing_cols:
    test_df[column] = 0  # Add missing dummy columns
test_df = test_df[required_columns]

# Predict using the saved model
model = load('random_forest_model.joblib')
test_df['isFraud'] = model.predict(test_df)

# Save the predictions
test_df.to_csv('test_predictions.csv', index=False)
