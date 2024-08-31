from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect form data
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        # Prepare the input array for the model
        input_features = np.array([[tv, radio, newspaper]])

        # Get the model's prediction
        prediction = model.predict(input_features)[0]

        return render_template('submit.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
