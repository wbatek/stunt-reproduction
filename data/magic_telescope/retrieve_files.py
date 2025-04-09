import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

def retrieve(seed):
    SEED = seed

    # Define the path where the files should be saved
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'magic_telescope'))

    # Create the directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)

    data = pd.read_csv(os.path.join(project_dir, "magic04.data"), header=None)

    for col in data.select_dtypes([np.object_]):
        data[col] = data[col].str.decode('utf-8')

    # Separate features and target variable
    y = data.iloc[:, -1]  # Labels are in the last column
    X = data.iloc[:, :-1]  # Features start from the first column

    # Encode class labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), [0, 1, 2, 3, 4]),
        ('scaler', MinMaxScaler(), [5, 6])
    ])

    X = preprocessor.fit_transform(X)

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    # Save datasets as NumPy arrays in the specified directory
    np.save(os.path.join(project_dir, "train_x.npy"), X_train)
    np.save(os.path.join(project_dir, "ytrain.npy"), y_train)
    np.save(os.path.join(project_dir, "val_x.npy"), X_val)
    np.save(os.path.join(project_dir, "pseudo_val_y.npy"), y_val)
    np.save(os.path.join(project_dir, "xtest.npy"), X_test)
    np.save(os.path.join(project_dir, "ytest.npy"), y_test)

    print(f"Files saved in {project_dir}")