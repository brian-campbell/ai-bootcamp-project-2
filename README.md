# ai-bootcamp-project-2

# Phishing URL Detection with Machine Learning

This project focuses on detecting phishing URLs using machine learning models such as Random Forest, LightGBM, and XGBoost. The goal is to accurately classify URLs as either legitimate or phishing based on various features extracted from the URLs.

### Table of Contents

* [**Introduction**](#Introduction)

* [**Dataset**](#Dataset)

* [**Models Used**](#Models-Used)

* [**Feature Engineering**](#Feature-Engineering)

* [**Evaluation Metrics**](#Evaluation-Metrics)

* [**Results**](#Results)

* [**Visualizations**](#Visualizations)

* [**Installation and Usage**](#Installation-and-Usage)

* [**Files**](#Files)

* [**Conclusion**](#Conclusion)


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

A sample screenshot of the full data file is included here for reference:
![Screenshot 2024-09-11 at 10 09 44 AM](https://github.com/user-attachments/assets/d5424eee-3b86-4d76-a908-40bee3d3e253)



### Models Used

Several machine learning models were implemented for URL classification:

1)  Random Forest: An ensemble method that constructs multiple decision trees for classification.

2)  LightGBM: A gradient boosting framework that is highly efficient and optimized for large datasets.

3)  XGBoost: Another boosting algorithm that focuses on performance and regularization to reduce overfitting.

### Model Performance:

* Random Forest: Achieved an accuracy of 97%.

* LightGBM: Achieved an accuracy of 92%.

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
![output](https://github.com/user-attachments/assets/6053a1bd-fe0a-47f7-bc14-94410993a1a2)![output](https://github.com/user-attachments/assets/6547b2a3-3cb5-4cd4-8739-90fc44334ab3)![output](https://github.com/user-attachments/assets/8b5af988-7cb4-4167-a91b-ec4b0a7e48f0)

    * Confusion Matrix: Displays the number of correct and incorrect classifications.

    * ROC Curves: Illustrates the trade-off between true positive and false positive rates.

### Example Visualization:

* Pie Chart: The distribution of legitimate and phishing URLs based on the dataset.
![output](https://github.com/user-attachments/assets/84974b51-bc2e-4d24-b809-b93fa8a73e2d)


### Installation and Usage

* Prerequisites:

    * Python 3.10.x or higher
      
    * Anaconda 24.4.x or higher

    * Jupyter Notebook

    * Required Python packages: scikit-learn, xgboost, lightgbm, matplotlib, pandas, optuna, shap, tabgan, seaborn, and numpy

### Steps:

1) Clone the repository:

        
        git@github.com:brian-campbell/ai-bootcamp-project-2.git
        cd ai-bootcamp-project-2

    
2) Run the model training and analysis:

        jupyter notebook phishing_basic_analysis.ipynb

5) Visualize the results: The notebook will generate various plots and graphs to visualize the results of the model training.

### Files

The main file for the project is `phishing_basic_analysis.ipynb` which is a jupyter notebook.

The main data file is `dataset_full.csv` which contains the main, large dataset for the project.

`Machine_Learning_of_Phishing_URL_Data.pptx` is the presentation that will be used for showing the class what was done in the project.

`filtered_dataset.csv` is a file that is written by the project. It contains a whittle down set of data from the main dataset file.

The `misc` directory contains several support files used during the proejct but are not part of the final solution.


### Conclusion

This project demonstrates the effectiveness of machine learning in detecting phishing URLs. Among the models tested, XGBoost proved to be the most reliable in terms of accuracy and performance. By analyzing features like URL length, SSL certificates, and domain characteristics, the models were able to distinguish phishing URLs from legitimate ones with a high degree of accuracy.

