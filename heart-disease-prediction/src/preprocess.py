import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path):
    """
    Loads heart disease dataset and performs preprocessing.
    Returns train-test split data.
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Handle missing values
    for col in X.columns:
        if X[col].dtype == "object":
            X[col].fillna(X[col].mode()[0], inplace=True)
        else:
            X[col].fillna(X[col].median(), inplace=True)

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
