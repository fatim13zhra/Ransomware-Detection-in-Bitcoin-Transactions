from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
xgboost_model = joblib.load('xgboost_model.pkl')

# Define route to render input form
@app.route('/')
def index():
    return render_template('index_predict.html')

# Define route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    length = float(request.form['length'])
    weight = float(request.form['weight'])
    count = int(request.form['count'])
    looped = int(request.form['looped'])
    neighbors = int(request.form['neighbors'])
    income = float(request.form['income'])
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'length': [length],
        'weight': [weight],
        'count': [count],
        'looped': [looped],
        'neighbors': [neighbors],
        'income': [income]
    })
    
    # Make predictions using the loaded XGBoost model
    prediction = xgboost_model.predict(input_data)[0]

    # Display the prediction on the result page
    return render_template('result_predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
