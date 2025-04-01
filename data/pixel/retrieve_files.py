import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.io import arff
import numpy as np
import pandas as pd


def retrieve(seed):
    SEED = seed

    # Define the path where the files should be saved
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pixel'))

    # Create the directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)

    # Load the ARFF file
    data, meta = arff.loadarff(os.path.join(project_dir, "pixel.arff"))

    # Convert structured array to NumPy array
    data_array = np.array(data.tolist(), dtype=object)

    # Separate features and target variable
    X = data_array[:, 1:]  # Features start from index 1 (excluding the first column)
    y = data_array[:, 0]  # Labels are in the first column

    # Decode target variable if it's in bytes
    if isinstance(y[0], bytes):
        y = np.array([label.decode('utf-8') for label in y])

    # Encode class labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(np.unique(y))

    # Scale numerical features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.astype(float))  # Ensure numeric conversion

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    # Save datasets as NumPy arrays in the specified directory
    np.save(os.path.join(project_dir, "train_x.npy"), X_train)
    np.save(os.path.join(project_dir, "val_x.npy"), X_val)
    np.save(os.path.join(project_dir, "xtest.npy"), X_test)
    np.save(os.path.join(project_dir, "ytrain.npy"), y_train)
    np.save(os.path.join(project_dir, "pseudo_val_y.npy"), y_val)
    np.save(os.path.join(project_dir, "ytest.npy"), y_test)

    print(f"Files saved in {project_dir}")
