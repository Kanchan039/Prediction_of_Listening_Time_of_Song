import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the training and testing data
train_data = pd.read_csv(r"C:\Users\Lenvo\Downloads\train_song.csv")  
test_data = pd.read_csv(r"C:\Users\Lenvo\Downloads\test_song.csv")   

# Separate features and target variable for training data
X_train = train_data.drop(columns=['Listening_Time_minutes', 'id'])
y_train = train_data['Listening_Time_minutes']

# Separate categorical and numerical columns for preprocessing
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(exclude=['object']).columns

# Create preprocessing pipeline for categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),  # Impute missing values for numerical columns
        ('cat', Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),  # OneHotEncode categorical columns
            ('imputer', SimpleImputer(strategy='most_frequent'))  # Impute missing values for categorical columns
        ]), categorical_cols)
    ])

# Create a full pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Prepare test data (apply the same preprocessing, without target variable)
X_test = test_data.drop(columns=['id'])

# Predict the Listening Time for the test set
y_pred = pipeline.predict(X_test)

# Prepare submission file with id and predicted Listening Time
test_data['Listening_Time_minutes'] = y_pred
submission = test_data[['id', 'Listening_Time_minutes']]

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)

# If you want to calculate RMSE for training data
y_train_pred = pipeline.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Training RMSE: {train_rmse}")
