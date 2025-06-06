# Heart_Disease_Prediction

Heart Disease Prediction Using Medallion Architecture and REST API
Table of Contents
Project Overview
Dataset
Medallion Architecture
Model Development
REST API Development
Deployment on Render
API Documentation
Testing Using Postman
Screenshots
Setup Instructions
Repository Structure
Future Enhancements

References
Project Overview
This project predicts the presence of heart disease in patients using the Heart Disease UCI dataset. It implements a Medallion Architecture on MongoDB for layered data processing and develops machine learning classification models trained on refined data. A REST API built with FastAPI serves model predictions, which is deployed on Render for public access.
Dataset
Source: Heart Disease UCI Dataset on Kaggle

URL: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

Description: Contains patient features such as age, sex, chest pain type, cholesterol levels, and a binary target indicating heart disease presence (0 = no, 1 = yes).

Medallion Architecture
Bronze Layer
Raw CSV data stored as JSON documents in MongoDB collection heart_disease_bronze.
No preprocessing applied at this stage.

Silver Layer
Handles missing values by imputing means for numerical and modes for categorical features.
Encodes categorical features numerically (label encoding or one-hot encoding).
Stored in heart_disease_silver collection.

Gold Layer
Normalizes numerical features using min-max scaling to range [0, 1].
Selects top features based on correlation and domain knowledge for model training.
Stored in heart_disease_gold collection.

Model Development
Used data from the Gold layer for training and testing.
Models trained: Logistic Regression, Random Forest, XGBoost, SVM.
Final model chosen based on highest accuracy, precision, recall, F1-score, and ROC-AUC metrics.
Feature selection based on correlation and feature importance from model outputs.
Hyperparameter tuning performed using Grid Search for optimal model performan
Data split: 80% training, 20% testing.
Final model saved as a .pkl file using joblib for deployment.

REST API Development
Built with FastAPI to serve model predictions.

Endpoints:
POST /predict: Accepts JSON patient data and returns heart disease prediction.
GET /health: Returns status to verify API is running.
Includes input validation to ensure valid and complete feature data.
Loads the saved model for real-time prediction.

Deployment on Render
API deployed on Render with environment variables configured for MongoDB connection.

Publicly accessible URL: https://heart-disease-prediction-tpn4.onrender.com
API Documentation
Swagger UI available at /docs for interactive API exploration (FastAPI built-in).

Endpoint details:
POST /predict
Input: JSON with patient features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal).
Output: JSON with prediction (0 or 1).

GET /health
Returns HTTP 200 OK to indicate service is up.
Testing Using Postman
Setup POST request with JSON body to /predict.

Example JSON input:
json
{
  "age": 60,
  "sex": 1,
  "cp": 0,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}

Response example
json

{
  "prediction": 1
}

Health check GET request to /health returns HTTP status 200.
Screenshots of Postman requests and responses included in the screenshots/ directory.
Screenshots
MongoDB Data: Bronze, Silver, and Gold collections showing sample documents.
Postman testing for /predict and /health endpoints.

Swagger API docs UI screenshot.
Setup Instructions
Clone the repository.
Install dependencies: pip install -r requirements.txt

Setup MongoDB Atlas and get the connection string.

Configure environment variables for MongoDB URI in .env file or directly in the Render dashboard for deployment.
5. Run API locally:  uvicorn main:app --reload
Use Postman or Swagger UI to test endpoints locally.

Repository Structure
bash
CopyEdit
├── data/                      # Raw and processed data files
├── notebooks/                 # Jupyter notebooks for data prep and modeling
├── api/                      # FastAPI source code
│   ├── main.py               # API entry point
│   ├── model.pkl             # Saved trained model
├── screenshots/              # MongoDB and Postman screenshots
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .env                      # Environment variables (not committed)

Future Enhancements
Add authentication and user roles for API.
Implement batch predictions and logging.
Integrate frontend UI for easier data input.
Expand model with more advanced feature engineering.

References
Heart Disease UCI Dataset: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
MongoDB Atlas: https://www.mongodb.com/cloud/atlas
FastAPI Documentation: https://fastapi.tiangolo.com/
Render Deployment: https://render.com
