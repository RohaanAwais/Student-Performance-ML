
from pathlib import Path

import json
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Student Performance Prediction API",
    description="Linear Regression and Logistic Regression models",
    version="1.0"
)

# Load saved models
regression_model = joblib.load(
    BASE_DIR / "regression_model.joblib"
)

classification_model = joblib.load(
    BASE_DIR / "classification_model.joblib"
)

# Load model evaluation results
with open(BASE_DIR / "metrics.json", encoding="utf-8") as file:
    metrics = json.load(file)

FEATURES = [
    "age",
    "avg_sleep_hours_weekday",
    "daily_screen_time_hours",
    "pre_sleep_screen_time_min",
    "caffeine_drinks_per_day",
    "sleep_quality_score",
    "exercise_days_per_week",
    "weekly_study_hours",
    "class_attendance_percent"
]


# Validate incoming student data
class StudentInput(BaseModel):
    age: float = Field(ge=17, le=27)
    avg_sleep_hours_weekday: float = Field(ge=1.5, le=10.5)
    daily_screen_time_hours: float = Field(ge=0.5, le=19.5)
    pre_sleep_screen_time_min: float = Field(ge=0, le=240)
    caffeine_drinks_per_day: float = Field(ge=0, le=7)
    sleep_quality_score: float = Field(ge=1, le=10)
    exercise_days_per_week: float = Field(ge=0, le=7)
    weekly_study_hours: float = Field(ge=0, le=88.5)
    class_attendance_percent: float = Field(ge=36.9, le=100)


@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "templates" / "index.html"
    )


@app.get("/metrics")
def get_metrics():
    return metrics


@app.post("/predict/regression")
def predict_regression(student: StudentInput):
    input_data = pd.DataFrame(
        [student.model_dump()],
        columns=FEATURES
    )

    prediction = float(
        regression_model.predict(input_data)[0]
    )

    return {
        "model": "Linear Regression",
        "predicted_cgpa": round(prediction, 3)
    }


@app.post("/predict/classification")
def predict_classification(student: StudentInput):
    input_data = pd.DataFrame(
        [student.model_dump()],
        columns=FEATURES
    )

    predicted_class = int(
        classification_model.predict(input_data)[0]
    )

    probabilities = classification_model.predict_proba(
        input_data
    )[0]

    classes = list(classification_model.classes_)
    high_probability = float(
        probabilities[classes.index(1)]
    )

    category = (
        "CGPA 3.0 or above"
        if predicted_class == 1
        else "CGPA below 3.0"
    )

    return {
        "model": "Logistic Regression",
        "predicted_class": predicted_class,
        "predicted_category": category,
        "high_cgpa_probability": round(high_probability, 4)
    }
