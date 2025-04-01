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
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'diabetes'))

    # Create the directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)

    # Load the ARFF file
    data, meta = arff.loadarff(os.path.join(project_dir, "diabetes.arff"))

    # Convert structured array to DataFrame
    df = pd.DataFrame(data)

    # Decode byte strings (if necessary)
    for col in df.select_dtypes([np.object_]):
        df[col] = df[col].str.decode('utf-8')

    # Separate features and target variable
    y = df.iloc[:, -1]  # Labels are in the last column
    X = df.iloc[:, :-1]  # Features start from the first column

    # Encode class labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Define preprocessing steps
    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols),
        # Drop first to avoid extra column
        ('scaler', MinMaxScaler(), numerical_cols)
    ])

    # Apply transformations
    X = preprocessor.fit_transform(X)

    # Split data into training, validation, and test sets
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
