from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pickled pipeline and model
with open('anemia_model.pkl', 'rb') as f:
    pipeline, model = pickle.load(f)

def convert_age_to_group(age):
    if 15 <= age <= 19:
        return '15-19'
    elif 20 <= age <= 24:
        return '20-24'
    elif 25 <= age <= 29:
        return '25-29'
    elif 30 <= age <= 34:
        return '30-34'
    elif 35 <= age <= 39:
        return '35-39'
    elif 40 <= age <= 44:
        return '40-44'
    elif 45 <= age <= 49:
        return '45-49'
    else:
        return 'unknown'
    
def get_recommendation(anemia_level):
    recommendations = {
        0: "Maintain a balanced diet rich in iron, vitamin B12, and folic acid. Consider regular checkups if you're at risk.",
        1: "Increase iron-rich foods like spinach, red meat, and lentils. Consider iron supplements if needed, after consulting a healthcare provider.",
        2: "Increase iron intake, and consult a healthcare provider to discuss possible iron or vitamin supplementation.",
        3: "Seek immediate medical consultation to identify the underlying cause and discuss treatment options like supplements or other interventions."
    }
    return recommendations.get(anemia_level, "No specific recommendation available.")

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for the general information page
@app.route('/general_info')
def general_info():
    return render_template('general_info.html')

# Route for the factors affecting anemia page
@app.route('/factors')
def factors():
    return render_template('factors.html')

# Route to display the prediction form
@app.route('/predict')
def predict():
    return render_template('prediction.html')

# Route to handle form submission, make predictions, and render results with recommendations
@app.route('/predict', methods=['POST'])
def make_prediction():
    # Map user input to the feature order in the DataFrame
    input_data = {
        'Age':  convert_age_to_group(int(request.form['Age'])),
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

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data using the pipeline
    processed_input = pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_input)

    # Ensure the prediction is an integer between 0 and 3
    prediction_value = int(prediction[0]) if isinstance(prediction[0], (int, np.int64)) else 0

    # Map the prediction output to a human-readable string
    prediction_map = {
        0: 'No Anemia',
        1: 'Mild Anemia',
        2: 'Moderate Anemia',
        3: 'Severe Anemia'
    }

    # Get the mapped prediction text
    prediction_text = prediction_map.get(prediction_value, 'Unknown')

    # Get the tailored recommendation based on the anemia level
    recommendation_text = get_recommendation(prediction_value)

    # Render the results page with the prediction and recommendations
    return render_template('result.html', prediction_text=f'Predicted Anemia Level: {prediction_text}', recommendation_text=recommendation_text)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
