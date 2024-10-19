from flask import Flask, jsonify, request
import pandas as pd
import pickle
import json  # Import json module for writing to file
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and label encoder from files
def load_model_and_encoder():
    try:
        with open('model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return best_model, label_encoder
    except FileNotFoundError:
        raise FileNotFoundError("Model or label encoder file not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the model or encoder: {e}")

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

# Create a Weather object from a dictionary
def create_weather_from_dict(data):
    try:
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
    except KeyError as e:
        raise ValueError(f"Missing key in input data: {e}")

@app.route('/evaluate', methods=['GET'])
def evaluate():
    try:
        model, label_encoder = load_model_and_encoder()

        # Dummy values for the sake of example
        accuracy = 75.0
        classification_report_dict = {
            'Drought': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75, 'support': 100},
            'Flood': {'precision': 0.6, 'recall': 0.65, 'f1-score': 0.62, 'support': 80},
            # Add more classes as needed
        }
        confusion_matrix_dict = {
            'labels': ['Drought', 'Flood'],
            'matrix': [[50, 10], [20, 60]]
        }

        result = {
            'accuracy': accuracy,
            'classification_report': classification_report_dict,
            'confusion_matrix': confusion_matrix_dict
        }

        # Write result to output.json
        with open('output.json', 'w') as f:
            json.dump(result, f, indent=4)

        return jsonify({'message': 'Evaluation results written to output.json'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        current_weather = create_weather_from_dict(data)
        current_weather_df = pd.DataFrame([current_weather.to_dict()])

        model, label_encoder = load_model_and_encoder()

        # Predict disaster type for current weather
        current_weather_encoded = model.predict(current_weather_df)
        predicted_disaster_type = label_encoder.inverse_transform(current_weather_encoded)[0]

        # Predict probabilities for current weather
        probabilities = model.predict_proba(current_weather_df)[0]
        disaster_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}

        # Filter probabilities
        filtered_probabilities = {disaster: prob for disaster, prob in disaster_probabilities.items() if prob >= 0.12}

        result = {
            'predicted_disaster_type': predicted_disaster_type,
            'disaster_probabilities': filtered_probabilities
        }

        # Write result to output.json
        with open('output.json', 'w') as f:
            json.dump(result, f, indent=4)

        return jsonify({'message': 'Prediction results written to output.json'})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
