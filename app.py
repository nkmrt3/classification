import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model("wine_points_model")  # Update with the correct path

# Function to preprocess the input text and make predictions
def predict_wine_review(text):
    text = [text]  # Convert input text to a list
    prediction = model.predict(text)[0][0]
    return "Good" if prediction >= 0.5 else "Bad"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        prediction = predict_wine_review(review)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
