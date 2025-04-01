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
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'karhunen'))

    # Create the directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)

    # Load the ARFF file
    data, meta = arff.loadarff(os.path.join(project_dir, "dataset_16_mfeat-karhunen.arff"))

    categorical_features = [i for i, attr in enumerate(meta) if meta[attr][0] == 'nominal']
    print(categorical_features)
    data_array = np.array(data.tolist(), dtype=object)

    X = data_array[:, :-1]
    y = data_array[:, -1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    # Save datasets as NumPy arrays in the specified directory
    np.save(os.path.join(project_dir, "train_x.npy"), X_train)
    np.save(os.path.join(project_dir, "val_x.npy"), X_val)
    np.save(os.path.join(project_dir, "xtest.npy"), X_test)
    np.save(os.path.join(project_dir, "ytrain.npy"), y_train)
    np.save(os.path.join(project_dir, "pseudo_val_y.npy"), y_val)
    np.save(os.path.join(project_dir, "ytest.npy"), y_test)

    print(f"Files saved in {project_dir}")
