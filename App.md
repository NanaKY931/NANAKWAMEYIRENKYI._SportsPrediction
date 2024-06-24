# app.py
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('best_model.pkl', 'rb'))

# Load the dataset for feature names and player statistics
players_22 = pd.read_csv('players_22.csv')

# Define the features used in the model
features = [
'overall', 'potential', 'value_eur', 'age', 'height_cm', 'pace'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    player_stats = {feature: data[feature] for feature in features}
    
    # Prepare the data for prediction
    input_data = pd.DataFrame([player_stats])
    
    # Perform prediction
    prediction = model.predict(input_data)
    
    return render_template('result.html', prediction=prediction, player_stats=player_stats)

if __name__ == '__main__':
    app.run(debug=True)
