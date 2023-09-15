import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your pre-trained model and Hub layer
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)
model = tf.keras.models.load_model('wine_points_model')

# Function to preprocess input text
def preprocess_input(description):
    description = [description]
    description = hub_layer(description)
    return description

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the wine description from the form
        wine_description = request.form['wine_description']

        # Preprocess the input
        preprocessed_input = preprocess_input(wine_description)

        # Make predictions using the model
        predicted_prob = model.predict(preprocessed_input)[0][0]

        # Determine the label based on a threshold (e.g., 0.5)
        predicted_label = 1 if predicted_prob >= 0.5 else 0

        # Prepare the prediction result to pass to the template
        prediction = {
            'probability': predicted_prob,
            'label': 'Yes' if predicted_label == 1 else 'No'
        }

        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
