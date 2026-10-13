python shap_explain.pyimport pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# New student
new_student = pd.DataFrame({
    "CGPA": [8.5],
    "Coding": [85],
    "Aptitude": [80],
    "Communication": [82],
    "Internship": [1]
})

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(new_student)

# Prediction
print("Prediction:", model.predict(new_student))

print("SHAP values:")
print(shap_values)

# Create SHAP graph
values = shap_values[0, :, 1]

plt.figure()
plt.bar(new_student.columns, values)
plt.xticks(rotation=45)
plt.title("SHAP Explanation - Student Placement Prediction")
plt.tight_layout()

# Save graph
plt.savefig("shap_explanation.png")
plt.close()

print("SHAP graph saved as shap_explanation.png")
