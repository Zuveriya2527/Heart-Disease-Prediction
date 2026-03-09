from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load ML model
model = pickle.load(open("models/heart_model.pkl", "rb"))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ---------------- HEART DISEASE PREDICTION ---------------- #

@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    bp = float(request.form['bp'])
    cholesterol = float(request.form['cholesterol'])
    smoke = int(request.form['smoke'])
    diabetes = int(request.form['diabetes'])
    activity = int(request.form['activity'])

    features = np.array([[age, bmi, bp, cholesterol, smoke, diabetes, activity]])

    prediction = model.predict(features)

    if prediction == 1:
        result = "⚠ High Risk of Heart Disease"
    else:
        result = "✅ Low Risk of Heart Disease"

    return render_template("dashboard.html", prediction=result)


@app.route('/story')
def story():
    return render_template('story.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/team')
def team():
    return render_template('team.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

