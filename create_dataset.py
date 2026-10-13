import pandas as pd
import random

random.seed(42)

students = []

for i in range(100):

    cgpa = round(random.uniform(5.0, 9.8), 1)
    coding = random.randint(40, 100)
    aptitude = random.randint(40, 100)
    communication = random.randint(50, 100)
    internship = random.randint(0, 1)

    score = (
        cgpa * 10
        + coding * 0.25
        + aptitude * 0.20
        + communication * 0.15
        + internship * 15
    )

    if score >= 135:
        placement = 1
    else:
        placement = 0

    students.append([
        cgpa,
        coding,
        aptitude,
        communication,
        internship,
        placement
    ])

data = pd.DataFrame(students, columns=[
    "CGPA",
    "Coding",
    "Aptitude",
    "Communication",
    "Internship",
    "Placement"
])

data.to_csv("student_data.csv", index=False)

print("100 student dataset created successfully!")
print(data)