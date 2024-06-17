from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model, scaler, and features
model_path = 'models/xgb_model1.pkl'
scaler_path = 'models/scaler.pkl1'
features_path = 'models/features.pkl'

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(features_path, 'rb') as features_file:
    features = pickle.load(features_file)

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Device Price Range Prediction API</h1>
    <p>This API predicts the price range of a device based on its specifications. 
    Send a POST request to the /predict endpoint with the device features in JSON format to get the price range prediction.</p>
    <p>Example JSON format:</p>
    <pre>
    {
      "battery_power": 842,
      "blue": 1,
      "clock_speed": 2.2,
      "dual_sim": 1,
      "fc": 1,
      "four_g": 1,
      "int_memory": 7,
      "m_dep": 0.6,
      "mobile_wt": 188,
      "n_cores": 2,
      "pc": 2,
      "px_height": 20,
      "px_width": 756,
      "ram": 2549,
      "sc_h": 9,
      "sc_w": 7,
      "talk_time": 19,
      "three_g": 0,
      "touch_screen": 0,
      "wifi": 1,
      "ram_per_px": 3.5,
      "battery_screen": 500
    }
    </pre>
    """

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify(features)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        input_data = [data.get(feature) for feature in features]
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Preprocess the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Return the result
        return jsonify({'price_range': int(prediction[0])})
    except ValueError as e:
        return jsonify({
            'error': str(e), 
            'message': 'Ensure all feature names match the training data.',
            'expected_features': features,
            'received_data': data
        }), 400
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
