# Breast Cancer Diagnosis Prediction using Machine Learning

This project implements a comprehensive machine learning pipeline to predict breast cancer diagnosis (malignant vs benign) using the Wisconsin Breast Cancer dataset. Multiple classifiers are trained and evaluated to identify the best performing model.

## 📊 Dataset
Source: Wisconsin Breast Cancer Dataset
Features: 30 numerical features derived from digitized images of breast mass biopsies
Target: Diagnosis (M = Malignant, B = Benign)
Total Samples: ~569 (after preprocessing)
                                                                                                                                                                
## 🛠️ Technologies Used

Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 📈 Pipeline Overview

1. **Data Loading & Exploration**
- Load CSV dataset
- Check shape, preview data, and data types
- Visualize diagnosis distribution
  
2. **Data Preprocessing**
- Remove columns with missing values
- Encode categorical target (M→1, B→0)
- Feature-target split (30 features)
- Train-test split (80-20)
- Standardization using StandardScaler
  
3. **Exploratory Data Analysis (EDA)**
- Count plot of diagnosis distribution
- Pair plot of key features
- Correlation heatmap (first 10 features)
  
4. **Machine Learning Models**
Model                   Parameters
Logistic Regression	- random_state=0
Decision Tree	        - criterion="entropy", random_state=0
Random Forest	        - n_estimators=10, criterion="entropy", random_state=0

5. **Model Evaluation**
-Training Accuracy: All models
-Testing Metrics: Classification Report + Accuracy Score
-Predictions: Random Forest predictions vs actual values


## 📁 Project Structure
breast-cancer-prediction/
│
├── Breast_Cancer.csv          # Dataset
├── breast_cancer_analysis.py  # Main analysis script
├── requirements.txt           # Dependencies
└── README.md                 # This file

## 🔬 Results Summary
Model Evaluation on Test Set:
- Logistic Regression: 92% accuracy
- Decision Tree: 88% accuracy  
- Random Forest: 94% accuracy (Best performing)
Key Findings: Random Forest Classifier achieved the highest accuracy, demonstrating robust performance on unseen data.


📝 Requirements File
numpy
pandas
matplotlib
seaborn
scikit-learn

