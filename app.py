import os
current_dir = os.path.dirname(os.path.abspath(__file__))


from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)
model_path = os.path.join(current_dir, 'model', 'forest_fire_model_lr.pkl')
scalarx_path = os.path.join(current_dir, 'model', 'scaler_X.pkl')
scaler_y_path = os.path.join(current_dir, 'model', 'scaler_y.pkl')

# Load the models and scalers from the 'models' folder
model_lr = pickle.load(open(model_path, 'rb'))
scaler_X = pickle.load(open(scalarx_path, 'rb'))
scaler_y = pickle.load(open(scaler_y_path, 'rb'))

# Load the dataset to calculate the mean values and maximum area
data_path = os.path.join(current_dir, 'dataset', 'new_df.csv')
dataset = pd.read_csv(data_path)
mean_values = dataset[['month', 'day', 'FFMC', 'ISI', 'rain', 'DMC_DC_combined', 'temp_wind_interaction']].mean().values
max_area = dataset['area'].max()

def predict_area_burned(temp, wind, RH):
    input_data = np.array([temp, wind, RH, *mean_values]).reshape(1, -1)
    input_data_scaled = scaler_X.transform(input_data)
    prediction_scaled = model_lr.predict(input_data_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
    normalized_prediction = prediction[0] / max_area
    return np.clip(normalized_prediction, 0, 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        temperature = float(data['temperature'])
        wind = float(data['wind'])
        RH = float(data['RH'])

        prediction = predict_area_burned(temperature, wind, RH)

        # Convert prediction to percentage and round it to 2 decimal places
        prediction_percentage = round(prediction * 100, 2)

        return jsonify({'probability': f"{prediction_percentage}%"})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred'}), 400


@app.route("/aboutus")
def about_us():
    return render_template('aboutus.html')

if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)))
