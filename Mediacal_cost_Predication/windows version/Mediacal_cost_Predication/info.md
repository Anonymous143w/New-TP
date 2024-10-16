# Medical Insurance Cost Prediction

## Overview
The **Medical Insurance Cost Prediction** project aims to develop a predictive model to estimate healthcare costs based on patient characteristics. Leveraging machine learning techniques, this project provides insights into the factors affecting medical expenses, enabling stakeholders to make informed decisions regarding healthcare services.

## Features
- **Data Ingestion and Preprocessing**: Automated data loading, preprocessing, and encoding of categorical variables to prepare for model training.
- **Predictive Modeling**: Implementation of a polynomial regression model for predicting medical costs based on various factors such as age, sex, BMI, number of children, smoking status, and region.
- **RESTful API**: A FastAPI-based web service that allows users to send data and receive predictions in real-time.
- **Model Evaluation**: Comprehensive evaluation metrics to assess model performance and accuracy.

## Technologies Used
- **Python**: The primary programming language used for data processing and model development.
- **Scikit-learn**: For machine learning algorithms, including label encoding and polynomial regression.
- **FastAPI**: To create a robust API for real-time predictions.
- **Pandas & NumPy**: For data manipulation and numerical calculations.
- **Joblib**: For saving and loading the trained model.
- **Ngrok**: For exposing the local server to the internet for demonstration purposes.
