from flask import Flask, jsonify, request
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

app = Flask(__name__)

# Define the Weather class
class Weather:
    def __init__(self, temp, feelsLike, pressure, humidity, clouds, visibility, wind, rain, snow, conditionId, main, description, icon):
        self.temp = temp
        self.feelsLike = feelsLike
        self.pressure = pressure
        self.humidity = humidity
        self.clouds = clouds
        self.visibility = visibility
        self.wind = wind
        self.rain = rain
        self.snow = snow
        self.conditionId = conditionId
        self.main = main
        self.description = description
        self.icon = icon

    def to_dict(self):
        return {
            'temp': self.temp,
            'feelsLike': self.feelsLike,
            'pressure': self.pressure,
            'humidity': self.humidity,
            'clouds': self.clouds,
            'visibility': self.visibility,
            'wind_deg': self.wind['deg'],
            'wind_gust': self.wind['gust'],
            'wind_speed': self.wind['speed'],
            'rain': self.rain,
            'snow': self.snow,
            'conditionId': self.conditionId,
            'main': self.main,
            'description': self.description
        }

# Load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Create a Weather object from a dictionary
def create_weather_from_dict(data):
    return Weather(
        temp=data['temp']['cur'],
        feelsLike=data['feelsLike']['cur'],
        pressure=data['pressure'],
        humidity=data['humidity'],
        clouds=data['clouds'],
        visibility=data['visibility'],
        wind=data['wind'],
        rain=data['rain'],
        snow=data['snow'],
        conditionId=data['conditionId'],
        main=data['main'],
        description=data['description'],
        icon=data['icon']
    )

# Prepare data for training
def prepare_data(disaster_data):
    features = []
    labels = []
    for d in disaster_data:
        weather_dict = d['weather']
        weather_flattened = create_weather_from_dict(weather_dict).to_dict()
        features.append(weather_flattened)
        labels.append(d['disasterType'])
    return pd.DataFrame(features), labels

# Load JSON files
def load_data():
    current_weather_data = load_json('current_weather.json')
    disaster_data = load_json('natural_disasters.json')
    return current_weather_data, disaster_data

# Train and evaluate the model
def train_and_evaluate_model():
    # Load and prepare data
    _, disaster_data = load_data()
    df, labels = prepare_data(disaster_data)

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Define numerical and categorical features
    numerical_features = ['temp', 'feelsLike', 'pressure', 'humidity', 'clouds', 'visibility', 'wind_deg', 'wind_gust', 'wind_speed', 'rain', 'snow', 'conditionId']
    categorical_features = ['main', 'description']

    # Define preprocessing for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Handle unknown categories
        ])

    # Create the pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Split data
    X = df
    y = encoded_labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model with hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_

    # Convert confusion matrix to dictionary
    def confusion_matrix_to_dict(cm, labels):
        return {
            'labels': list(labels),
            'matrix': cm.tolist()
        }

    # Convert classification report to dictionary
    def classification_report_to_dict(report):
        report_dict = {}
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                report_dict[label] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                    'support': metrics['support']
                }
        return report_dict

    # Save the model and label encoder
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    return accuracy, classification_report_to_dict(report), confusion_matrix_to_dict(cm, labels)

# Load the trained model and label encoder
def load_model_and_encoder():
    with open('model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return best_model, label_encoder

# Train and evaluate the model once when the app starts
accuracy, classification_report_dict, confusion_matrix_dict = train_and_evaluate_model()

@app.route('/evaluate', methods=['GET'])
def evaluate():
    result = {
        'accuracy': accuracy * 100,
        'classification_report': classification_report_dict,
        'confusion_matrix': confusion_matrix_dict
    }
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    current_weather = create_weather_from_dict(data)
    current_weather_df = pd.DataFrame([current_weather.to_dict()])

    # Load trained model and encoder
    best_model, label_encoder = load_model_and_encoder()

    # Predict disaster type for current weather
    current_weather_encoded = best_model.predict(current_weather_df)
    predicted_disaster_type = label_encoder.inverse_transform(current_weather_encoded)[0]

    # Predict probabilities for current weather
    probabilities = best_model.predict_proba(current_weather_df)[0]
    disaster_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

    filtered_probabilities = {disaster: prob for disaster, prob in disaster_probabilities.items() if prob >= 0.12}

    result = {
        'predicted_disaster_type': predicted_disaster_type,
        'disaster_probabilities': filtered_probabilities
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
