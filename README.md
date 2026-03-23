# 🩺 Diagnosing the Future: ML Approach to Predicting Diabetes

[![Language](https://img.shields.io/badge/Language-R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)
[![ML](https://img.shields.io/badge/Type-Machine%20Learning-orange?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)]()

## 📌 Project Overview

This project applies **Machine Learning techniques** to predict the likelihood of diabetes in patients using clinical and demographic data. Early and accurate detection of diabetes can significantly improve patient outcomes and reduce healthcare costs.

## 🎯 Objectives

- Build a predictive model to classify whether a patient is diabetic or not
- - Evaluate multiple ML algorithms and compare their performance
  - - Identify the most significant features contributing to diabetes risk
   
    - ## 🛠️ Technologies Used
   
    - | Tool/Library | Purpose |
    - |---|---|
    - | **R** | Primary programming language |
    - | **caret** | Model training and evaluation |
    - | **ggplot2** | Data visualization |
    - | **dplyr** | Data manipulation |
    - | **randomForest** | Random Forest classifier |
    - | **rpart** | Decision Tree classifier |
   
    - ## 📊 Dataset
   
    - - **Source:** Pima Indians Diabetes Dataset
      - - **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        - - **Target:** Diabetes outcome (0 = No, 1 = Yes)
         
          - ## 🔬 Methodology
         
          - 1. **Data Preprocessing** — Handling missing values, normalization, outlier detection
            2. 2. **Exploratory Data Analysis (EDA)** — Distribution plots, correlation heatmaps
               3. 3. **Model Training** — Logistic Regression, Decision Tree, Random Forest
                  4. 4. **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
                     5. 5. **Feature Importance** — Identifying key predictors of diabetes
                       
                        6. ## 📈 Results
                       
                        7. - Achieved high prediction accuracy using ensemble methods
                           - - Glucose level and BMI were identified as the most influential predictors
                             - - Random Forest outperformed other classifiers in cross-validation
                              
                               - ## 🚀 How to Run
                              
                               - ```r
                                 # Clone the repository
                                 git clone https://github.com/charish1307/BUSINESS-ANALYTICS-AND-DATA-MINING-Project.git

                                 # Open final project.R in RStudio
                                 install.packages(c("caret", "ggplot2", "dplyr", "randomForest", "rpart"))
                                 source("final project.R")
                                 ```

                                 ## 👤 Author

                                 **Charish Yadavali** | [GitHub](https://github.com/charish1307) | [LinkedIn](https://www.linkedin.com/in/charishyadavali)

                                 ---
                                 *University of Massachusetts Dartmouth — Business Analytics & Data Mining*
