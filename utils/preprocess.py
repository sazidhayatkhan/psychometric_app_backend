# utils/preprocess.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Preprocess the data
def preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Separate features (responses) and labels
    X = df.drop("Label", axis=1)  # Features (questions)
    y = df["Label"]  # Labels (course recommendations)

    # One-hot encode the features (responses)
    encoder = OneHotEncoder(sparse=False)
    X_encoded = encoder.fit_transform(X)

    # Convert encoded features back to a DataFrame for readability
    encoded_columns = encoder.get_feature_names_out(X.columns)
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)

    # Label encode the target (course labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save the encoders for later use
    joblib.dump(encoder, "models/one_hot_encoder.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    return X_encoded_df, y_encoded, encoder, label_encoder

# Train the model
def train_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": report
    }
    joblib.dump(metrics, "models/model_metrics.pkl")

    # Save the trained model
    joblib.dump(model, "models/course_recommendation_model.pkl")
    print("Model and metrics saved successfully")
    
    return metrics