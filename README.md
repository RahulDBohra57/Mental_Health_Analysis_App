🧠 Mental Health Cluster Insight Tool

A Machine Learning-based emotional well-being assessment app

📌 Project Summary

This project uses machine learning to help users understand their emotional and behavioral well-being. Based on eight self-reported indicators (such as stress, habits, mood swings, and social engagement), the system predicts a user’s mental-wellness cluster and provides supportive, therapist-style recommendations.

The tool is built using a combination of:

Multiple Correspondence Analysis (MCA) for transforming categorical responses

K-Means Clustering for identifying well-being profiles

Streamlit for the interactive web interface

ReportLab for generating downloadable PDF wellness reports

The output includes:

A Cluster Identification: one of six emotional-wellbeing profiles

A Psychological Interpretation: describing patterns in behaviors and stress responses

Personalized Suggestions: practical recommendations to support mental wellness

A PDF Report: containing user responses, cluster insight, date/time, and guidance

💡 How It Works (Short Version)

User selects responses to 8 mental-wellness questions.

Responses are passed into the trained pipeline:

MCA converts categories (Yes/No/Maybe) into numerical factors.

Scaler standardizes the numerical features.

K-Means assigns the user to one of six psychological clusters.

The cluster is mapped to a professionally-written interpretation and suggestions.

User can download a personalized Wellness Report (PDF).

🏗 Model Training

Model training was completed in Jupyter Notebook:

Data cleaning → exploratory analysis

MCA transformation

K-Means clustering

Cluster profiling

Exporting model artifacts (joblib) for deployment

🚀 Deployment

The Streamlit application loads the trained MCA transformer, scaler, and clustering model, collects user inputs, generates predictions, and provides insights and reports.
