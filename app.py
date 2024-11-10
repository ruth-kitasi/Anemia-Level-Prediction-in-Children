from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pickled pipeline and model
with open('anemia_model.pkl', 'rb') as f:
    pipeline, model = pickle.load(f)

# Define the feature names in the same order as they were used during training
feature_names = [
    'Age', 'Residence', 'Highest educational level', 'Wealth index',
    'Births in last five years', 'Age of respondent at 1st birth',
    'Hemoglobin level', 'Have mosquito net', 'marital status',
    'Residing with partner', 'Had fever in last two weeks', 'Taking iron medication'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Map user input to the feature order in the DataFrame
    input_data = {
        'Age': request.form['Age'],
        'Residence': request.form['Residence'],
        'Highest educational level': request.form['Highest educational level'],
        'Wealth index': request.form['Wealth index'],
        'Births in last five years': request.form['Births in last five years'],
        'Age of respondent at 1st birth': request.form['Age of respondent at 1st birth'],
        'Hemoglobin level': request.form['Hemoglobin level'],
        'Have mosquito net': request.form['Have mosquito net'],
        'marital status': request.form['marital status'],
        'Residing with partner': request.form['Residing with partner'],
        'Had fever in last two weeks': request.form['Had fever in last two weeks'],
        'Taking iron medication': request.form['Taking iron medication']
    }
    
    # Convert the input data to a DataFrame with the correct feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Preprocess the input data
    processed_input = pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_input)

    # Render the prediction
    return render_template('index.html', prediction_text=f'Predicted Anemia Level: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
