from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    player_data = {
        'feature1': request.form['feature1'],
        'feature2': request.form['feature2'],
        'feature3': request.form['feature3'],
        # Add more features here
    }
    
    input_data = pd.DataFrame(player_data, index=[0])
    prediction = model.predict(input_data)
    player_data['rating'] = prediction[0]
    
    return render_template('result.html', player_data=player_data)

if __name__ == '__main__':
    app.run(debug=True)
