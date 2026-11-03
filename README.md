# Customer Churn Prediction System

A machine learning web application that predicts customer churn using Python, Scikit-learn, Random Forest and Streamlit.

## Project Overview

This project analyzes telecom customer data to identify customers who are likely to discontinue their services. It provides churn predictions and an interactive business analytics dashboard to help understand customer behavior.

## Features

* Customer churn prediction using a trained Random Forest model.
* Churn probability for individual customers.
* Interactive business analytics dashboard.
* Customer retention and churn statistics.
* Contract-wise churn analysis.
* Feature importance visualization.
* Dataset preview and CSV download.

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* Random Forest
* Streamlit
* Matplotlib and Seaborn
* Jupyter Notebook

## Project Structure

```text
Customer-Churn-Prediction/
├── app.py
├── requirements.txt
├── dataset/
├── models/
├── notebooks/
└── .gitignore
```

## Run Locally

1. Clone this repository:

   `git clone https://github.com/AshishPoojary20/Customer-Churn-Prediction.git`

2. Navigate to the project directory.

3. Install dependencies:

   `pip install -r requirements.txt`

4. Run the application:

   `python -m streamlit run app.py`

5. Open the local URL displayed in your terminal.

## Machine Learning Model

The project uses a Random Forest classifier trained on telecom customer data to predict customer churn.

## Author

Ashish Poojary
