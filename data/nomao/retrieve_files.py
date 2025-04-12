import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.io import arff
import numpy as np
import pandas as pd

def retrieve(seed):
    SEED = seed

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'nomao'))

    os.makedirs(project_dir, exist_ok=True)

    data, meta = arff.loadarff(os.path.join(project_dir, "nomao.arff"))
    df = pd.DataFrame(data)

    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('scaler', MinMaxScaler(), numerical_cols)
    ])

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)
    print(X_train.shape)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()

    # Save datasets as NumPy arrays in the specified directory
    np.save(os.path.join(project_dir, "train_x.npy"), X_train)
    np.save(os.path.join(project_dir, "val_x.npy"), X_val)
    np.save(os.path.join(project_dir, "xtest.npy"), X_test)
    np.save(os.path.join(project_dir, "ytrain.npy"), y_train)
    np.save(os.path.join(project_dir, "pseudo_val_y.npy"), y_val)
    np.save(os.path.join(project_dir, "ytest.npy"), y_test)

    print(f"Files saved in {project_dir}")
retrieve(0)