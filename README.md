# Predicting Subscription Churn Using Sentiment Analysis of Customer Support Interactions

## Overview

This project aims to predict customer churn for a subscription service by analyzing the sentiment expressed in customer support interactions.  The analysis leverages natural language processing (NLP) techniques to gauge customer satisfaction and identify at-risk subscribers.  By classifying the sentiment (positive, negative, neutral) of customer support interactions, the model helps predict the likelihood of churn and allows for proactive intervention strategies to improve retention rates.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script:** Execute the main script using:

   ```bash
   python main.py
   ```

   This will perform the sentiment analysis, churn prediction, and generate the output.


## Example Output

The script will print key analysis results to the console, including:

* Summary statistics of the sentiment distribution across customer interactions.
* Accuracy metrics of the churn prediction model (e.g., precision, recall, F1-score).
*  A summary of the predicted churn probability for each customer (if applicable).

Additionally, the script will generate the following plot files in the `output` directory:

* `sentiment_distribution.png`: A bar chart visualizing the distribution of positive, negative, and neutral sentiments.
* `churn_prediction_performance.png`: A plot illustrating the performance of the churn prediction model (e.g., ROC curve or confusion matrix).  *(Note:  The specific plot will depend on the chosen evaluation metric).*


## Data

The project requires a dataset containing customer support interaction text and corresponding churn labels.  This dataset is assumed to be provided in a CSV file named `customer_interactions.csv` located in the project's root directory.  The expected format of this file is detailed in `data_format.md`.  (Please update this section with your data format description).

## License

[Specify your license here, e.g., MIT License]