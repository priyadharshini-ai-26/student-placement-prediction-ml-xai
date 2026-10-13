from flask import Flask, request, render_template_string, send_file
import pickle
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


html = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Placement Prediction</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 650px;
            margin: auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 25px;
        }

        .header h1 {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .card {
            background: white;
            padding: 30px;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        label {
            display: block;
            text-align: left;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 6px;
        }

        input {
            width: 100%;
            padding: 13px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
        }

        button {
            width: 100%;
            padding: 14px;
            margin-top: 25px;
            border: none;
            border-radius: 8px;
            background: #667eea;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        .result {
            margin-top: 25px;
            padding: 18px;
            border-radius: 10px;
            background: #f1f5f9;
            font-size: 20px;
            font-weight: bold;
        }

        .shap-section {
            margin-top: 30px;
            text-align: center;
        }

        .shap-image {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #ddd;
        }

        .footer {
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1>🎓 Student Placement Prediction</h1>
        <p>Machine Learning Based Placement Prediction System</p>
    </div>

    <div class="card">

        <form method="POST">

            <label>CGPA</label>
            <input type="number"
                   step="0.1"
                   name="cgpa"
                   placeholder="Enter CGPA"
                   required>

            <label>Coding Score</label>
            <input type="number"
                   name="coding"
                   placeholder="Enter Coding Score"
                   required>

            <label>Aptitude Score</label>
            <input type="number"
                   name="aptitude"
                   placeholder="Enter Aptitude Score"
                   required>

            <label>Communication Score</label>
            <input type="number"
                   name="communication"
                   placeholder="Enter Communication Score"
                   required>

            <label>Internship</label>
            <input type="number"
                   min="0"
                   max="1"
                   name="internship"
                   placeholder="Enter 0 or 1"
                   required>

            <button type="submit">
                Predict Placement
            </button>

        </form>

        {% if result %}

        <div class="result">
            {{ result }}
        </div>

        <div class="shap-section">
            <h2>📊 SHAP Explanation for This Student</h2>

            <img class="shap-image"
                 src="/shap_explanation.png">
        </div>

        {% endif %}

    </div>

    <div class="footer">
        Student Placement Prediction | Decision Tree + SHAP
    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        cgpa = float(request.form["cgpa"])
        coding = int(request.form["coding"])
        aptitude = int(request.form["aptitude"])
        communication = int(request.form["communication"])
        internship = int(request.form["internship"])

        # Student data entered by user
        student = pd.DataFrame({
            "CGPA": [cgpa],
            "Coding": [coding],
            "Aptitude": [aptitude],
            "Communication": [communication],
            "Internship": [internship]
        })

        # Prediction
        prediction = model.predict(student)

        if prediction[0] == 1:
            result = "✅ Student is likely to be PLACED!"
        else:
            result = "❌ Student is likely NOT to be PLACED."

        # Create SHAP explanation for THIS student
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(student)

        # Handle SHAP output format
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        # If SHAP returns 3 dimensions: samples, features, classes
        if hasattr(values, "ndim") and values.ndim == 2:
            values = values[:, 1]

        # Create graph
        plt.figure(figsize=(8, 5))

        plt.bar(student.columns, values)

        plt.xticks(rotation=30)
        plt.ylabel("SHAP Value")
        plt.title("SHAP Explanation for Current Student")

        plt.tight_layout()

        # Save updated graph
        plt.savefig("shap_explanation.png")
        plt.close()

    return render_template_string(html, result=result)


@app.route("/shap_explanation.png")
def shap_image():

    return send_file("shap_explanation.png")


if __name__ == "__main__":
    app.run(debug=True)
    