# Student Performance Prediction System

## Project Overview
A machine learning web application that predicts student CGPA and classifies students into CGPA categories using sleep, screen-time, lifestyle, and academic features.

## Machine Learning Models
- **Linear Regression:** Predicts numerical CGPA.
- **Logistic Regression:** Predicts whether CGPA is below 3.0 or at least 3.0.

The models are developed using scikit-learn and saved with joblib.

## Evaluation
- Regression: R², MAE, and RMSE.
- Classification: Accuracy, precision, recall, and F1-score.

## Technologies
Python, pandas, NumPy, scikit-learn, joblib, FastAPI, Uvicorn, HTML, CSS, and JavaScript.

## Dataset
Student Sleep, Screen Time & Academic Performance.

Source: https://www.kaggle.com/datasets/aamir28/student-sleep-screen-time-and-academic-performance

## Project Structure
- `ML_Assignment.ipynb` — data exploration, training, and evaluation.
- `app.py` — FastAPI backend and prediction endpoints.
- `templates/index.html` — HTML frontend.
- `regression_model.joblib` — saved regression pipeline.
- `classification_model.joblib` — saved classification pipeline.
- `metrics.json` — evaluation metrics.
- `requirements.txt` — Python dependencies.

## How to Run

1. Install Python.
2. Create and activate a virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Open `ML_Assignment.ipynb` in VS Code and run all cells to train and save the models.
5. Start the API:

   `python -m uvicorn app:app --reload`

6. Open `http://127.0.0.1:8000` to use the application.
7. Open `http://127.0.0.1:8000/docs` to test the API.

## API Endpoints
- `POST /predict/regression`
- `POST /predict/classification`

Both endpoints accept student feature values as JSON and return predictions as JSON.

## Classification Definition
CGPA below 3.0 is class 0; CGPA of 3.0 or above is class 1. The classification labels are derived from the dataset's CGPA column.

## Limitations
The predictions are estimates and do not establish causal relationships between lifestyle factors and academic performance.

