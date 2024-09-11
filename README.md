# ai-bootcamp-project-2

# Phishing URL Detection with Machine Learning

This project focuses on detecting phishing URLs using machine learning models such as Random Forest, LightGBM, and XGBoost. The goal is to accurately classify URLs as either legitimate or phishing based on various features extracted from the URLs.

### Table of Contents

* Introduction

* Dataset

* Models Used

* Feature Engineering

* Evaluation Metrics

* Results

* Visualizations

* Installation and Usage

* Conclusion


### Introduction

Phishing is a major threat to internet users, where malicious actors mimic legitimate websites to steal sensitive information. The goal of this project is to leverage machine learning models to identify and classify phishing URLs by analyzing their features.

### Key Objectives:

* Use various machine learning models to classify phishing and legitimate URLs.

* Evaluate the performance of different models to determine the best one.

* Visualize key results and model performances.

### Dataset

The dataset used in this project contains a collection of legitimate and phishing URLs. Key features in the dataset include:

* URL length  

* Presence of special characters

* Domain age

* SSL certificate status

* Redirection flags

These features are crucial in determining whether a URL is phishing or legitimate.

There dataset contains 112 columns of data and just over 88,000 rows of data. It was obtained here: https://github.com/GregaVrbancic/Phishing-Dataset


## Models Used

Several machine learning models were implemented for URL classification:

1)  Random Forest: An ensemble method that constructs multiple decision trees for classification.

2)  LightGBM: A gradient boosting framework that is highly efficient and optimized for large datasets.

3)  XGBoost: Another boosting algorithm that focuses on performance and regularization to reduce overfitting.

### Model Performance:

* Random Forest: Achieved an accuracy of 97%.

* LightGBM: Achieved an accuracy of 94%.

* XGBoost: Achieved the highest accuracy of 97%.

### Feature Engineering

Feature engineering involved:

 * Extracting and generating meaningful features from URLs, such as the presence of specific characters and SSL certificates.

 * Handling missing values by either imputing or removing them.

 * Scaling and normalizing numerical features where necessary.

### Evaluation Metrics

To evaluate the performance of the models, the following metrics were used:

* Accuracy: The percentage of correct predictions.

* Precision: Measures the accuracy of phishing detections.

* Recall: Measures the model’s ability to detect all phishing URLs.

* F1-Score: The harmonic mean of precision and recall.

* AUC-ROC: Measures the ability of the model to distinguish between classes.

### Model Comparison:

The models were compared using the above metrics, and XGBoost emerged as the most accurate, balancing precision and recall effectively.

### Results

* Random Forest: This model provided good results but was prone to overfitting, leading to slightly lower accuracy.

* LightGBM: Offered a faster training time and performed well with large datasets.

* XGBoost: Performed the best overall, balancing accuracy and speed while avoiding overfitting through regularization.

### Visualizations

* The following visualizations were generated as part of the analysis:

    * Feature Importance Plots: Show the contribution of each feature to the model’s decision.

    * Confusion Matrix: Displays the number of correct and incorrect classifications.

    * ROC Curves: Illustrates the trade-off between true positive and false positive rates.

### Example Visualization:

* Pie Chart: The distribution of legitimate and phishing URLs based on the dataset.

### Installation and Usage

* Prerequisites:

    * Python 3.x

    * Jupyter Notebook

    * Required Python packages: scikit-learn, xgboost, lightgbm, matplotlib, pandas

### Steps:

1) Clone the repository:

        git clone https://github.com/username/phishing-url-detection.git
        cd phishing-url-detection

    
 2) Install dependencies:

        pip install -r requirements.txt

3) Run the model training and analysis:

        jupyter notebook phishing_basic_analysis.ipynb

4) Visualize the results: The notebook will generate various plots and graphs to visualize the results of the model training.

### Conclusion

This project demonstrates the effectiveness of machine learning in detecting phishing URLs. Among the models tested, XGBoost proved to be the most reliable in terms of accuracy and performance. By analyzing features like URL length, SSL certificates, and domain characteristics, the models were able to distinguish phishing URLs from legitimate ones with a high degree of accuracy.

