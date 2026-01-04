# Student Performance Prediction - ML Pipeline

A comprehensive machine learning pipeline for predicting student exam scores using multiple regression models with robust validation and sanity checks.

## Overview

This project implements a production-ready ML pipeline that:
- Performs extensive data quality checks
- Trains and compares multiple regression models
- Uses proper cross-validation strategies
- Identifies potential data leakage and synthetic artifacts
- Provides feature importance analysis

## Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

## Dataset

The pipeline expects a CSV file named `student_performance_prediction_dataset.csv` with the following features:

**Numerical Features:**
- `age` - Student age
- `study_hours` - Weekly study hours
- `class_attendance` - Attendance percentage
- `assignment_submission_rate` - Assignment completion rate
- `failures` - Number of past failures
- `parent_education` - Parent education level (ordinal)
- `sleep_hours` - Average sleep hours
- `sleep_quality` - Sleep quality rating
- `social_life_index` - Social activity index
- `facility_rating` - School facility rating

**Categorical Features:**
- `gender` - Student gender
- `course` - Course enrolled
- `internet_access` - Internet availability
- `study_method` - Primary study method
- `exam_difficulty` - Exam difficulty level

**Target Variable:**
- `exam_score` - Exam score (0-100)

## Pipeline Architecture

### Step 1: Dataset Sanity Check
- Missing value detection
- Duplicate row identification
- Target variable range validation (0-100)
- Constant column removal

### Step 2: Feature Typing
- Automatic separation of numerical and categorical features
- ID column removal to prevent data leakage

### Step 3: EDA & Risk Assessment
- Correlation analysis with target variable
- Detection of suspiciously high correlations (>0.95)
- Skewness analysis for synthetic artifact detection
- Feature distribution checks

### Step 4: Cross-Validation Strategy
- 5-Fold Cross-Validation with shuffling
- Prevents overfitting and provides robust performance estimates

### Step 5: Preprocessing
Two preprocessing pipelines are implemented:

**Linear Models (Ridge/Lasso):**
- StandardScaler for numerical features
- OneHotEncoder for categorical features

**Tree Models (XGBoost/LightGBM):**
- Passthrough for numerical features (no scaling needed)
- OrdinalEncoder for categorical features

### Steps 6-9: Model Training & Evaluation

Four models are trained and compared:

1. **Ridge Regression** - Baseline linear model with L2 regularization
2. **Lasso Regression** - Feature selection via L1 regularization
3. **LightGBM** - Gradient boosting with optimized parameters
4. **XGBoost** - Alternative gradient boosting for confirmation


## Output

The pipeline provides:

1. **Data Quality Report**
   - Missing values count
   - Duplicate detection
   - Target variable validation

2. **Correlation Analysis**
   - Top 5 positive correlations
   - Top 5 negative correlations
   - Data leakage warnings

3. **Model Performance (RMSE)**
   - Cross-validated scores for all models
   - Standard deviation of CV scores

4. **Feature Importance**
   - Top 10 most important features (from LightGBM)
   - Sanity checks for feature dominance

5. **Lasso Pruning Report**
   - Number of coefficients set to zero
   - Feature selection effectiveness

## Model Hyperparameters

### Ridge & Lasso
- Ridge alpha: 1.0
- Lasso alpha: 0.01 (small to avoid over-pruning)

### LightGBM
- n_estimators: 1000
- learning_rate: 0.05
- max_depth: 10
- num_leaves: 31

### XGBoost
- n_estimators: 1000
- learning_rate: 0.05
- max_depth: 6
