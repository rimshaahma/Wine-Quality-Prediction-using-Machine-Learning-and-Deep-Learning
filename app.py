from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('wine_quality_model.h5')

# Define scaler (same as used during training)
scaler = StandardScaler()
data = pd.read_csv('dataset/winequality.csv').drop(columns=['quality', 'Id'])
scaler.fit(data)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    try:
        features = request.json
        input_data = np.array([[
            features['fixed acidity'],
            features['volatile acidity'],
            features['citric acid'],
            features['residual sugar'],
            features['chlorides'],
            features['free sulfur dioxide'],
            features['total sulfur dioxide'],
            features['density'],
            features['pH'],
            features['sulphates'],
            features['alcohol']
        ]])
        
        # Preprocess input
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        quality = prediction[0][0]  # Regression output
        
        return jsonify({'quality': float(quality)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
