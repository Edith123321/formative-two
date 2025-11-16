# Formative Two - Product Recommendation System

## Project Overview
A machine learning-based product recommendation system that analyzes customer transaction history and social media engagement to deliver personalized product suggestions.

## Features
- **Data Processing**: Handles missing values, feature engineering, and data normalization
- **Model Training**: Implements XGBoost and RandomForest classifiers with hyperparameter tuning
- **Evaluation**: Comprehensive metrics (accuracy, F1-score, precision, recall)
- **Visualization**: Generates confusion matrices and feature importance plots

# Dependencies
      
## Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/formative-two.git](https://github.com/yourusername/formative-two.git)
   cd formative-two

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Usage
1. Run the training pipeline:
    ```bash
    python -m product_recommendation_system.src.train

2. Evaluate the model:
    ```bash
    python -m product_recommendation_system.src.evaluate   

# Results
Test Accuracy: 16.67%
## Class Distribution:
Class 0: 20.00% of samples
Class 1: 22.00% of samples
Class 2: 27.00% of samples
Class 3: 23.00% of samples
Class 4: 28.00% of samples

## Data Processing
Missing Values: Handled by filling missing customer ratings with the median value (3.0)

Feature Engineering:
    Extracted date components (year, month, day) from purchase dates
    Processed numerical and categorical features separately
Data Splitting: 80-20 train-test split with stratification

## Model Architecture
Baseline Model: HistGradientBoostingClassifier (20.00% accuracy)
Main Model: XGBoost with class weights to handle class imbalance

Class Weights:
    Class 0: 1.20
    Class 1: 1.09
    Class 2: 0.89
    Class 3: 1.04
    Class 4: 0.86

## Evaluation
Metrics:
    Accuracy: 16.67%
    Classification report with precision, recall, and F1-score
    Confusion matrix visualization
Visualizations:
    Class distribution
    Feature importance
    Confusion matrix

# Team Members
- Edith Githinji – Data Analyst  
- Wenebifid – Data Scientist  
- Patricia Mugabo – ML Engineer
- Queen WIHOGORA - Data Engineer