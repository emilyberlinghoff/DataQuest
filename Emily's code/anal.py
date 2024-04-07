import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()

# Custom function to handle unseen labels during transformation
def safe_transform(column, trained_encoder):
    unseen_label = -1  # Assign -1 for unseen labels
    return column.map(lambda x: trained_encoder.transform([x])[0] if x in trained_encoder.classes_ else unseen_label)

# Initialize LabelEncoders
label_encoders = {}

# Load the dataset
df = pd.read_csv('train_cleaned.csv')

def preprocess_datetime(df):
    """
    Preprocess datetime columns of the DataFrame.
    """
    df['transDate'] = pd.to_datetime(df['transDate'])
    df['year'] = df['transDate'].dt.year
    df['month'] = df['transDate'].dt.month
    df['day'] = df['transDate'].dt.day
    return df.drop('transDate', axis=1)

def preprocess_categorical(df, label_encoders=None, fit_encoders=False):
    """
    Encode categorical variables. If fit_encoders is True, fit and return label encoders.
    """
    categorical_cols = ['category', 'city', 'job', 'state', 'business', 'firstName', 'lastName', 'street']
    
    if fit_encoders:
        label_encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        if col in df.columns:  # Check if the column exists
            if fit_encoders:
                df[col] = label_encoders[col].fit_transform(df[col])
            else:
                df[col] = safe_transform(df[col], label_encoders[col])
    
    # One-hot encoding for 'gender' if it exists
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    
    return df, label_encoders if fit_encoders else df

def annotate_reasons(df, feature_importances, features):
    """
    Annotate each row in df with the top 3 reasons contributing to its classification.
    """
    top_features_indices = np.argsort(feature_importances)[-3:]  # Get indices of top 3 features
    top_features = features[top_features_indices]
    reasons = ', '.join(top_features)
    df['SuspiciousReasons'] = np.where(df['isFraud'] == 1, reasons, '')

# Convert the date/time column to datetime and extract components
df = preprocess_datetime(df)

# Encoding categorical variables and one-hot encoding for gender
df, label_encoders = preprocess_categorical(df, label_encoders=label_encoders, fit_encoders=True)

# Separate features and target variable
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42, verbose=2)
model.fit(X_train, y_train)

# Predict on the test set and display performance
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model and encoders
dump(model, 'random_forest_model.joblib')
for col, le in label_encoders.items():
    dump(le, f'{col}_encoder.joblib')

# Load new data, preprocess
test_df = pd.read_csv('test.csv')
test_df = preprocess_datetime(test_df)
test_df, _ = preprocess_categorical(test_df, label_encoders=label_encoders, fit_encoders=False)

# Align features of test_df to match those of X_train (the training features)
test_df_aligned = test_df.reindex(columns=X_train.columns, fill_value=0)

# Add empty column fo suspicious reasons
test_df['SuspiciousReasons'] = ''

# Predict using the saved model and annotate reasons based on feature importances
test_df['isFraud'] = model.predict(test_df_aligned)
annotate_reasons(test_df, model.feature_importances_, X_train.columns)

# Save the predictions with annotations
test_df.to_csv('test_predictions_with_reasons.csv', index=False)

# Align test_df to match the training features, adding missing columns with 0s
aligned_test_df = pd.DataFrame(columns=X_train.columns)
for column in aligned_test_df.columns:
    if column in test_df.columns:
        aligned_test_df[column] = test_df[column]
    else:
        aligned_test_df[column] = 0  # Adding missing columns with default value

# Selecting only the columns present in the training set
X_test = test_df[X_train.columns]

# Ensure to only select the columns used for training when predicting
predictions = model.predict(aligned_test_df)

# Add predictions back to test_df for annotation
test_df['isFraud'] = predictions

# Save the predictions
test_df.to_csv('test_predictions.csv', index=False)
