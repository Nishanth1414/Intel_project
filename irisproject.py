import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = '/home/test4/dataset/iris.csv'  
iris_df = pd.read_csv(file_path)

# Inspect the dataset
print(iris_df.head())

# Separate features and labels
X = iris_df.iloc[:, :-1].values  # All columns except the last
y = iris_df.iloc[:, -1].values   # Last column (species)

# Encode categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Convert species names into numerical labels

# Standardize the features for better SVM performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different SVM kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
for kernel in kernels:
    # Train SVM model
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM with {kernel} kernel Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
