import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the players_22 dataset
players_22 = pd.read_csv('players_22.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract player features from the form
    player_features = [float(x) for x in request.form.values()]
    player_features = np.array(player_features).reshape(1, -1)
    
    # Predict rating
    prediction = model.predict(player_features)
    rating = prediction[0]
    
    # Calculate confidence interval
    preds = []
    for estimator in model.estimators_:
        preds.append(estimator.predict(player_features))
    preds = np.array(preds)
    std_dev = np.std(preds)
    confidence_level = 95
    z_score = 1.96  # for 95% confidence level
    lower_bound = rating - z_score * std_dev
    upper_bound = rating + z_score * std_dev

    # Get player information (assuming the form contains player_id)
    player_id = request.form.get('player_id')
    player_info = players_22.loc[players_22['player_id'] == int(player_id)].to_dict(orient='records')[0]
    
    return render_template('result.html', rating=rating, lower_bound=lower_bound, upper_bound=upper_bound, confidence_level=confidence_level, player_info=player_info)

if __name__ == '__main__':
    app.run(debug=True)
