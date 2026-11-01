# Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy multiple machine learning classification models to predict whether a song will go "viral." By analyzing various audio features (such as danceability, energy, etc.), the models classify songs into binary categories (viral or not viral). The final application is deployed via a Streamlit interactive web interface for real-time model evaluation on test data.

## b. Dataset Description
* **Dataset Name:** Viral Songs Dataset (`viral_songs.csv`)
* **Total Instances:** 10,000 (Meets the >500 minimum requirement)
* **Total Features:** 17 features (Meets the >12 minimum requirement)
* **Target Variable:** `viral` (Binary: 0 for non-viral, 1 for viral)
* **Description:** The dataset contains tracks with their corresponding audio characteristics. Non-predictive text features like track name, artist, and track ID were removed during preprocessing, and the categorical `genre` feature was label-encoded.

## c. Github Repository Link
https://github.com/AzeemIbrahim/Viral-Song-Classifier

## d. Models used:
The following 5 classification models were trained and evaluated on the dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

### Evaluation Metrics Comparison Table
*(Note: Metrics were calculated on the 20% test split of the dataset)*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8120 | 0.8884 | 0.8010 | 0.6366 | 0.7094 | 0.5814 |
| **Decision Tree** | 0.7610 | 0.7411 | 0.6681 | 0.6699 | 0.6690 | 0.4820 |
| **KNN** | 0.7095 | 0.7610 | 0.6118 | 0.5312 | 0.5687 | 0.3533 |
| **Naive Bayes** | 0.7800 | 0.8665 | 0.9072 | 0.4341 | 0.5872 | 0.5199 |
| **Support Vector Machine** | 0.7960 | 0.8703 | 0.8075 | 0.5700 | 0.6683 | 0.5439 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall accuracy (81.2%) and AUC (0.888), showing strong, balanced predictive capability across both classes. |
| **Decision Tree** | Displayed the most balanced Precision and Recall scores (~0.66), though overall accuracy lagged behind linear models. |
| **KNN** | Performed the poorest on this dataset, yielding the lowest metrics across the board, including a weak MCC of 0.353. |
| **Naive Bayes** | Yielded exceptional Precision (90.7%) but very poor Recall (43.4%), indicating it is highly conservative when predicting a track is viral. |
| **Support Vector Machine** | Delivered robust overall performance as a strong runner-up to Logistic Regression, with high AUC (0.870) and good accuracy. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the clear winner. It achieved the highest Accuracy, AUC, F1 Score, and MCC, demonstrating the most reliable and balanced performance for this dataset. |
