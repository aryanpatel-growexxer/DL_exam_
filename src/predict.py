import pandas as pd
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn


test_df_raw = pd.read_csv('test.csv')
print("test.csv loaded successfully.")



def preprocess_data_fixed(df_raw, preprocessing_pipeline_path='preprocessing_pipeline.pkl'):
    """
    Applies the saved preprocessing steps to a raw DataFrame, with fixes for encoding inconsistencies.
    """
    # Load the preprocessing components
    with open(preprocessing_pipeline_path, 'rb') as f:
        components = pickle.load(f)

    gender_encoder = components['gender_encoder']
    insurance_type_encoder = components['insurance_type_encoder']
    mean_age = components['mean_age']
    gender_glucose_median = components['gender_glucose_median']

    df = df_raw.copy()

    columns_to_drop = ['patient_id', 'admission_date', 'discharge_day_of_week']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    def safe_encode(df, column, encoder):
        """Helper to handle encoders fitted on ints vs strings."""
        if column in df.columns:
            try:
                # Try to transform a string sample to see if encoder accepts strings
                _ = encoder.transform(pd.Series([str(df[column].iloc[0])]))
                df[column] = encoder.transform(df[column].astype(str))
            except (ValueError, TypeError):
                print(f"Warning: '{column}_encoder' seems incompatible with strings. Re-fitting a temporary LabelEncoder.")
                temp_enc = LabelEncoder()
                df[column] = temp_enc.fit_transform(df[column].astype(str))
        return df

    # Apply Label Encoding with fixes for both columns
    df = safe_encode(df, 'gender', gender_encoder)
    df = safe_encode(df, 'insurance_type', insurance_type_encoder)

    if 'age' in df.columns:
        df['age'] = df['age'].replace(999, np.nan)
        df['age'].fillna(mean_age, inplace=True)

    if 'glucose_level_mgdl' in df.columns and df['glucose_level_mgdl'].isnull().any():
        if 'gender' in df.columns:
            # Ensure gender is numeric for mapping
            if df['gender'].dtype == 'object':
                df['glucose_level_mgdl'].fillna(df['glucose_level_mgdl'].median(), inplace=True)
            else:
                df['glucose_level_mgdl'] = df['glucose_level_mgdl'].fillna(df['gender'].map(gender_glucose_median))
        else:
            df['glucose_level_mgdl'].fillna(df['glucose_level_mgdl'].median(), inplace=True)

    global X_train
    if 'X_train' in globals() and isinstance(X_train, pd.DataFrame):
        missing_cols = set(X_train.columns) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        df = df[X_train.columns]
    else:
        print("Warning: X_train not found in global scope. Cannot ensure column order.")

    return df

preprocessed_test_df = preprocess_data_fixed(test_df_raw, preprocessing_pipeline_path='preprocessing_pipeline.pkl')

# Define the architecture (MUST match the trained model keys from the error)
class SimpleANN(nn.Module):
    def __init__(self, input_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Re-instantiate the model with the updated architecture
input_size = preprocessed_test_df.shape[1]
loaded_model = SimpleANN(input_size).to(device)

# Load the saved state dictionary
model_save_path = 'simple_ann_model.pth'
try:
    if torch.cuda.is_available():
        loaded_model.load_state_dict(torch.load(model_save_path))
    else:
        loaded_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    print("Model loaded successfully with updated architecture.")
except RuntimeError as e:
    print(f"Architecture mismatch persisted: {e}")

loaded_model.eval() # Set to evaluation mode

print("Loaded Model Architecture:")
print(loaded_model)

# Make predictions
print("\nMaking predictions...")

# Convert preprocessed test data to PyTorch tensor
test_tensor = torch.tensor(preprocessed_test_df.values, dtype=torch.float32).to(device)

with torch.no_grad():
    raw_outputs = loaded_model(test_tensor)
    # Apply the same threshold as used in evaluation (0.7)
    binary_predictions = (raw_outputs > 0.7).float().cpu().numpy()

# Convert binary predictions to 'Yes'/'No'
final_predictions = pd.Series(binary_predictions.flatten()).map({1.0: 'Yes', 0.0: 'No'})

# Create a simplified output DataFrame with patient_id and prediction
output_df = pd.DataFrame({
    'patient_id': test_df_raw['patient_id'],
    'readmitted_30d_prediction': final_predictions.values
})


# Save the output DataFrame to a CSV file
output_csv_path = 'predicted.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"Predictions successfully saved to {output_csv_path}")
