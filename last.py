# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# File Paths
DATASET_PATH = "C:/Users/abdul/OneDrive/Desktop/sybil_address_prediction/sybil_address_prediction/"
transactions_file = DATASET_PATH + "transactions.parquet"
token_transfers_file = DATASET_PATH + "token_transfers.parquet"
dex_swaps_file = DATASET_PATH + "dex_swaps.parquet"
train_addresses_file = DATASET_PATH + "train_addresses.parquet"
test_addresses_file = DATASET_PATH + "test_addresses.parquet"

# Load Data
transactions = pd.read_parquet(transactions_file)
token_transfers = pd.read_parquet(token_transfers_file)
dex_swaps = pd.read_parquet(dex_swaps_file)
train_addresses = pd.read_parquet(train_addresses_file)
test_addresses = pd.read_parquet(test_addresses_file)

# Convert Column Names to Uppercase
transactions.columns = transactions.columns.str.upper()
token_transfers.columns = token_transfers.columns.str.upper()
dex_swaps.columns = dex_swaps.columns.str.upper()
train_addresses.columns = train_addresses.columns.str.upper()
test_addresses.columns = test_addresses.columns.str.upper()

# Merge DataFrames
merged_df = transactions.copy()
if 'TX_HASH' in token_transfers.columns:
    merged_df = pd.merge(merged_df, token_transfers, on='TX_HASH', how='outer', suffixes=('_TX', '_TT'))
if 'TX_HASH' in dex_swaps.columns:
    merged_df = pd.merge(merged_df, dex_swaps, on='TX_HASH', how='outer', suffixes=('', '_DEX'))

# Handle NaN Values
datetime_cols = merged_df.select_dtypes(include=['datetime64[ns, UTC]']).columns
merged_df[datetime_cols] = merged_df[datetime_cols].fillna(pd.Timestamp('1970-01-01 00:00:00+0000', tz='UTC'))
merged_df.fillna(0, inplace=True)

# Rename and Consolidate Address Columns
merged_df.rename(columns={
    'FROM_ADDRESS_TX': 'SENDER_ADDRESS',
    'FROM_ADDRESS_TT': 'SENDER_ADDRESS',
    'FROM_ADDRESS': 'SENDER_ADDRESS'
}, inplace=True)

# Drop Duplicate Columns
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Check for Address Columns
address_columns = ['SENDER_ADDRESS', 'TO_ADDRESS_TX', 'TO_ADDRESS_TT', 'CONTRACT_ADDRESS']
available_address_columns = [col for col in address_columns if col in merged_df.columns]

# Label Encoding for Address Columns
label_encoders = {}
for col in available_address_columns:
    print(f"Encoding Column: {col}")
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col].astype(str))
    label_encoders[col] = le

# Convert both columns to string for consistent merging
train_addresses['ADDRESS'] = train_addresses['ADDRESS'].astype(str)
merged_df['SENDER_ADDRESS'] = merged_df['SENDER_ADDRESS'].astype(str)

# Merge with Labels
train_data = pd.merge(train_addresses, merged_df, left_on='ADDRESS', right_on='SENDER_ADDRESS', how='left')
train_data.fillna(0, inplace=True)

# Separate Features and Target
X = train_data.drop(columns=['ADDRESS', 'LABEL'])
y = train_data['LABEL']

# Resampling with SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass Distribution After Resampling:")
print(pd.Series(y_resampled).value_counts())

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Calculate scale_pos_weight Safely
pos_count = sum(y_train == 1)
neg_count = sum(y_train == 0)
if pos_count == 0 or neg_count == 0:
    scale_pos_weight = 1  # Fallback Value
else:
    scale_pos_weight = np.sqrt(neg_count / pos_count)

# Model: XGBoost Classifier with Hyperparameter Tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)

# Hyperparameters Grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid Search with Stratified K-Fold
grid_search = GridSearchCV(xgb, param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1', verbose=2)
grid_search.fit(X_train, y_train)

# Best Model
best_model = grid_search.best_estimator_
print("\nBest Hyperparameters:", grid_search.best_params_)

# Validation
y_pred = best_model.predict(X_val)
print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nAverage F1 Score:", f1_score(y_val, y_pred))

# Prepare Test Data
test_addresses['ADDRESS'] = test_addresses['ADDRESS'].astype(str)
test_data = pd.merge(test_addresses, merged_df, left_on='ADDRESS', right_on='SENDER_ADDRESS', how='left')
test_data.fillna(0, inplace=True)
X_test = test_data.drop(columns=['ADDRESS'])

# Predictions on Test Data
test_predictions = best_model.predict(X_test)

# Create Submission CSV
submission = pd.DataFrame({
    'ADDRESS': test_addresses['ADDRESS'],
    'PRED': test_predictions
})
submission.to_csv('sybil_predictions.csv', index=False)

print("\nSubmission file 'sybil_predictions.csv' created successfully.")
