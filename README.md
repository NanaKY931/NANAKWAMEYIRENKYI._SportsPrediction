from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

#Create the app

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


<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Player Rating Prediction</title>
</head>
<body>
    <h1>Player Rating Prediction</h1>
    <form action="{{ url_for('predict') }}" method="post">
        <label for="feature1">Feature 1:</label>
        <input type="text" id="feature1" name="feature1" required><br><br>
        
        <label for="feature2">Feature 2:</label>
        <input type="text" id="feature2" name="feature2" required><br><br>
        
        <label for="feature3">Feature 3:</label>
        <input type="text" id="feature3" name="feature3" required><br><br>
        
        <!-- Add more features here -->
        
        <input type="submit" value="Predict Rating">
    </form>

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Player Rating Result</title>
</head>
<body>
    <h1>Player Rating Result</h1>
    <table>
        <tr>
            <th>Feature</th>
            <th>Value</th>
        </tr>
        {% for key, value in player_data.items() %}
        <tr>
            <td>{{ key }}</td>
            <td>{{ value }}</td>
        </tr>
        {% endfor %}
    </table>
    <br>
    <a href="{{ url_for('home') }}">Predict Another Rating</a>
</body>
</html>
